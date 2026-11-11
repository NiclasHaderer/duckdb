#include "duckdb/common/vector/constant_vector.hpp"
#include "duckdb/common/vector/flat_vector.hpp"
#include "duckdb/common/vector/list_vector.hpp"
#include "duckdb/common/vector/map_vector.hpp"
#include "duckdb/common/vector/shredded_vector.hpp"
#include "duckdb/common/vector/variant_vector.hpp"
#include "duckdb/common/vector/struct_vector.hpp"
#include "duckdb/common/types/variant_iterator.hpp"
#include "duckdb/function/scalar/variant_utils.hpp"
#include "duckdb/function/scalar/variant_functions.hpp"
#include "duckdb/function/scalar/regexp.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/storage/statistics/struct_stats.hpp"

namespace duckdb {

VariantExtractBindData::VariantExtractBindData(const string &str) : FunctionData(), component(str) {
}
VariantExtractBindData::VariantExtractBindData(uint32_t index) : FunctionData() {
	if (index == 0) {
		throw BinderException("Extracting index 0 from VARIANT(ARRAY) is invalid, indexes are 1-based");
	}
	component = VariantPathComponent(index - 1);
}

unique_ptr<FunctionData> VariantExtractBindData::Copy() const {
	return make_uniq<VariantExtractBindData>(*this);
}

bool VariantExtractBindData::Equals(const FunctionData &other) const {
	auto &bind_data = other.Cast<VariantExtractBindData>();
	return component == bind_data.component;
}

static unique_ptr<BaseStatistics> VariantExtractPropagateStats(ClientContext &context, FunctionStatisticsInput &input) {
	auto &child_stats = input.child_stats;
	auto &bind_data = input.bind_data;

	auto &info = bind_data->Cast<VariantExtractBindData>();
	auto &variant_stats = child_stats[0];
	const bool is_shredded = VariantStats::IsShredded(variant_stats);
	if (!is_shredded) {
		return nullptr;
	}
	auto &shredded_stats = VariantStats::GetShreddedStats(variant_stats);
	if (!VariantShreddedStats::IsFullyShredded(shredded_stats)) {
		return nullptr;
	}
	auto found_stats = VariantShreddedStats::FindChildStats(shredded_stats, info.component);
	if (!found_stats || !VariantShreddedStats::IsFullyShredded(*found_stats)) {
		return nullptr;
	}

	auto result = VariantStats::WrapExtractedFieldAsVariant(variant_stats, *found_stats);
	if (info.component.lookup_mode == VariantChildLookupMode::BY_INDEX) {
		// Element statistics do not account for NULLs from out-of-bounds array indexes.
		result->SetHasNull();
	}
	return result;
}

static unique_ptr<FunctionData> VariantExtractBind(BindScalarFunctionInput &input) {
	auto &arguments = input.GetArguments();

	if (arguments.size() != 2) {
		throw BinderException("'variant_extract' expects two arguments, VARIANT column and VARCHAR path");
	}
	const auto &path = *arguments[1];
	if (path.GetReturnType().id() != LogicalTypeId::VARCHAR && !path.GetReturnType().IsIntegral()) {
		throw BinderException(
		    "'variant_extract' expects the second argument to be of type VARCHAR or any integer type, not %s",
		    path.GetReturnType().ToString());
	}

	auto constant_arg = input.GetNonNullConstant(1);

	if (constant_arg.type().id() == LogicalTypeId::VARCHAR) {
		return make_uniq<VariantExtractBindData>(constant_arg.GetValue<string>());
	} else if (constant_arg.type().IsIntegral()) {
		return make_uniq<VariantExtractBindData>(constant_arg.GetValue<uint32_t>());
	} else {
		throw InternalException("Constant-folded argument was not of type VARCHAR or any integer type");
	}
}

static void ExtractFromLeftover(const Vector &node, const Vector &unshredded,
                                const vector<VariantPathComponent> &components, idx_t path_index, Vector &result,
                                idx_t count);

//! References the leaf wrapper into 'result'. Extract returns a leftover VARIANT_NULL as SQL NULL, but the wrapper
//! would resolve it to a valid VARIANT_NULL root, so those rows get a NULL untyped_value_index: a NULL typed_value and
//! untyped_value_index resolve to SQL NULL. The wrapper itself stays valid, as a NULL struct cannot have valid
//! children.
static void ReferenceLeafWrapper(const Vector &input, const Vector &unshredded, Vector &result, idx_t count) {
	auto &input_entries = StructVector::GetEntries(input);
	D_ASSERT(input_entries.size() == 2);
	auto &input_untyped = input_entries[VariantStats::UNTYPED_VALUE_INDEX];
	auto typed_validity = input_entries[VariantStats::TYPED_VALUE_INDEX].Validity();
	auto untyped_indexes = input_untyped.Values<uint32_t>();
	unique_ptr<UnshreddedVariantIterator> leftovers;
	vector<idx_t> variant_null_rows;
	for (idx_t row = 0; row < count; row++) {
		auto untyped_index = untyped_indexes[row];
		if (typed_validity.IsValid(row) || !untyped_index.IsValid() || untyped_index.GetValue() == 0) {
			continue;
		}
		if (!leftovers) {
			leftovers = make_uniq<UnshreddedVariantIterator>(unshredded);
		}
		if (leftovers->GetTypeId(row, untyped_index.GetValue() - 1) == VariantLogicalType::VARIANT_NULL) {
			variant_null_rows.push_back(row);
		}
	}
	if (variant_null_rows.empty()) {
		result.Reference(input);
		return;
	}
	auto &result_entries = StructVector::GetEntries(result);
	result_entries[VariantStats::TYPED_VALUE_INDEX].Reference(input_entries[VariantStats::TYPED_VALUE_INDEX]);
	auto &result_untyped = result_entries[VariantStats::UNTYPED_VALUE_INDEX];
	VectorOperations::Copy(input_untyped, result_untyped, count, 0, 0);
	for (auto row : variant_null_rows) {
		FlatVector::SetNull(result_untyped, row, true);
	}
	auto input_validity = input.Validity();
	auto &validity = FlatVector::ValidityMutable(result);
	for (idx_t row = 0; row < count; row++) {
		if (!input_validity.IsValid(row)) {
			validity.SetInvalid(row);
		}
	}
}

static bool TryShreddedExtractRecursive(const Vector &input, const Vector &unshredded,
                                        const vector<VariantPathComponent> &components, Vector &result, idx_t count,
                                        idx_t path_index = 0) {
	if (path_index >= components.size()) {
		// reached the end of the path - shred
		auto &leaf_type = input.GetType();
		const bool is_wrapper = leaf_type.id() == LogicalTypeId::STRUCT;
		D_ASSERT(!is_wrapper || StructType::GetChildName(leaf_type, VariantStats::TYPED_VALUE_INDEX) == "typed_value");
		auto &typed_type =
		    is_wrapper ? StructType::GetChildType(leaf_type, VariantStats::TYPED_VALUE_INDEX) : leaf_type;
		if (typed_type.IsNested()) {
			// we only support this for non-nested types for now
			return false;
		}
		Vector shredded_vector(LogicalType::STRUCT({{"unshredded", unshredded.GetType()}, {"shredded", leaf_type}}),
		                       count);
		auto &top_shredded = StructVector::GetEntries(shredded_vector);
		auto &unshredded_child = top_shredded[0];
		auto &shredded_child = top_shredded[1];
		if (is_wrapper) {
			// the untyped_value_index of a STRUCT(typed_value, untyped_value_index) leaf points into the unshredded
			// part of the same row, so referencing the input's unshredded part keeps those rows resolvable
			unshredded_child.Reference(unshredded);
			ReferenceLeafWrapper(input, unshredded, shredded_child, count);
		} else {
			for (auto &unshredded_entry : StructVector::GetEntries(unshredded_child)) {
				ConstantVector::SetNull(unshredded_entry, count_t(count));
			}
			ConstantVector::SetNull(unshredded_child, count_t(count));
			shredded_child.Reference(input);
		}

		FlatVector::SetSize(shredded_vector, count_t(count));
		result.Shred(shredded_vector, count);
		return true;
	}
	auto &component = components[path_index];
	if (component.lookup_mode != VariantChildLookupMode::BY_KEY) {
		// only by key supported
		return false;
	}
	if (input.GetType().id() != LogicalTypeId::STRUCT) {
		//! Not shredded on OBJECT, can't extract a key
		return false;
	}
	// first entry is "typed_value"
	auto &typed_entries = StructVector::GetEntries(input);
	if (typed_entries.empty()) {
		// An empty shredded STRUCT has no typed_value child.
		return false;
	}
	auto &typed_value = typed_entries[0];
	if (typed_value.GetType().id() != LogicalTypeId::STRUCT) {
		// A shredded wrapper can also contain a primitive or an array.
		return false;
	}

	// find the type in the struct type
	auto &child_types = StructType::GetChildTypes(typed_value.GetType());
	auto &child_entries = StructVector::GetEntries(typed_value);
	for (idx_t child_idx = 0; child_idx < child_types.size(); child_idx++) {
		auto &entry = child_types[child_idx];
		if (entry.first.GetIdentifierName() == component.key) {
			// key found - move onto next component
			return TryShreddedExtractRecursive(child_entries[child_idx], unshredded, components, result, count,
			                                   path_index + 1);
		}
	}
	// a shredded key never also appears in the unshredded part of its object, so an unshredded key can only be in the
	// leftover object this node's untyped_value_index points to
	ExtractFromLeftover(input, unshredded, components, path_index, result, count);
	return true;
}

static bool TryFromShreddedExtract(const Vector &variant_vec, const vector<VariantPathComponent> &components,
                                   Vector &result, idx_t count) {
	if (variant_vec.GetVectorType() != VectorType::SHREDDED_VECTOR) {
		// input vector is not shredded
		return false;
	}
	auto &shredded_vec = ShreddedVector::GetShreddedVector(variant_vec);
	auto &unshredded_vec = ShreddedVector::GetUnshreddedVector(variant_vec);
	return TryShreddedExtractRecursive(shredded_vec, unshredded_vec, components, result, count);
}

//! Extracts 'components' from 'first_component' onwards out of the unshredded 'variant_vec', starting each valid row at
//! the value its 'value_index_sel' entry points to
static void ExtractFromValues(const Vector &variant_vec, const vector<VariantPathComponent> &components,
                              idx_t first_component, SelectionVector &value_index_sel, ValidityMask &validity,
                              Vector &result, idx_t count) {
	RecursiveUnifiedVectorFormat source_format;
	Vector::RecursiveToUnifiedFormat(variant_vec, source_format);

	UnifiedVariantVectorData variant(source_format);

	SelectionVector new_value_index_sel;
	new_value_index_sel.Initialize(count);

	const auto owned_nested_data = make_unsafe_uniq_array_uninitialized<VariantNestedData>(count);
	array_ptr nested_data(owned_nested_data.get(), count);

	//! Perform the extract
	const idx_t path_length = components.size() - first_component;
	for (idx_t i = 0; i < path_length; i++) {
		auto &component = components[first_component + i];
		auto &input_indices = i % 2 == 0 ? value_index_sel : new_value_index_sel;
		auto &output_indices = i % 2 == 0 ? new_value_index_sel : value_index_sel;

		auto expected_type = component.lookup_mode == VariantChildLookupMode::BY_INDEX ? VariantLogicalType::ARRAY
		                                                                               : VariantLogicalType::OBJECT;

		(void)VariantUtils::CollectNestedData(variant, expected_type, input_indices, count, optional_idx(), 0,
		                                      nested_data, validity);
		//! Look up the value_index of the child we're extracting
		ValidityMask lookup_validity(count);
		VariantUtils::FindChildValues(variant, component, nullptr, output_indices, lookup_validity, nested_data,
		                              validity, count);

		for (idx_t j = 0; j < count; j++) {
			if (!validity.RowIsValid(j)) {
				continue;
			}
			if (lookup_validity.CanHaveNull() && !lookup_validity.RowIsValid(j)) {
				//! No child could be extracted, set to NULL
				validity.SetInvalid(j);
				continue;
			}
			//! Get the index into 'values'
			auto type_id = variant.GetTypeId(j, output_indices[j]);
			if (type_id == VariantLogicalType::VARIANT_NULL) {
				validity.SetInvalid(j);
			}
		}
	}

	//! We reference the input, creating a dictionary over the 'values' list entry to remap what value_index 0 points
	//! to. This way we can perform a zero-copy extract on the variant column (when there are no nulls). The following
	//! code looks complicated but is necessary to avoid modifying the 'input'

	auto &values = UnifiedVariantVector::GetValues(source_format);
	auto values_data = values.GetData<list_entry_t>(values);
	auto &raw_values = VariantVector::GetValues(variant_vec);
	auto values_list_size = ListVector::GetListSize(raw_values);

	//! Create a new Variant that references the existing data of the input Variant
	result.Initialize(VectorDataInitialization::UNINITIALIZED, count);
	VariantVector::GetKeys(result).Reference(VariantVector::GetKeys(variant_vec));
	VariantVector::GetChildren(result).Reference(VariantVector::GetChildren(variant_vec));
	VariantVector::GetData(result).Reference(VariantVector::GetData(variant_vec));

	//! Copy the existing 'values' list entry data
	auto &result_values = VariantVector::GetValues(result);
	result_values.Initialize(VectorDataInitialization::UNINITIALIZED, count);
	ListVector::Reserve(result_values, values_list_size);
	ListVector::SetListSize(result_values, values_list_size);
	auto result_data = FlatVector::Writer<list_entry_t>(result_values, count);
	for (idx_t i = 0; i < count; i++) {
		if (!validity.RowIsValid(i)) {
			result_data.WriteNull();
			continue;
		}
		result_data.WriteValue(values_data[values.sel->get_index(i)]);
	}

	auto &result_indices = path_length % 2 == 0 ? value_index_sel : new_value_index_sel;

	//! Prepare the selection vector to remap index 0 of each row
	auto new_sel = SelectionVector::Incremental(values_list_size);
	for (idx_t i = 0; i < count; i++) {
		if (!validity.RowIsValid(i)) {
			continue;
		}
		auto &list_entry = values_data[values.sel->get_index(i)];
		new_sel.set_index(list_entry.offset, list_entry.offset + result_indices[i]);
	}

	auto &result_type_id = VariantVector::GetValuesTypeId(result);
	auto &result_byte_offset = VariantVector::GetValuesByteOffset(result);

	result_type_id.Slice(VariantVector::GetValuesTypeId(variant_vec), new_sel, values_list_size);
	result_byte_offset.Slice(VariantVector::GetValuesByteOffset(variant_vec), new_sel, values_list_size);

	if (validity.CanHaveNull()) {
		//! Create a copy of the vector, because we used Reference before, and we now need to adjust the data
		//! Which is a problem if we're still sharing the memory with 'input'
		Vector other(result.GetType(), count);
		VectorOperations::Copy(result, other, count, 0, 0);
		result.Reference(other);

		for (idx_t i = 0; i < count; i++) {
			if (!validity.RowIsValid(i)) {
				FlatVector::SetNull(result, i, true);
			}
		}
	}
	FlatVector::SetSize(result, count_t(count));
}

static void ExtractFromLeftover(const Vector &node, const Vector &unshredded,
                                const vector<VariantPathComponent> &components, idx_t path_index, Vector &result,
                                idx_t count) {
	auto &node_entries = StructVector::GetEntries(node);
	SelectionVector value_index_sel(count);
	ValidityMask validity(count);
	bool has_leftover = false;
	if (node_entries.size() > VariantStats::UNTYPED_VALUE_INDEX) {
		auto node_validity = node.Validity();
		auto untyped_indexes = node_entries[VariantStats::UNTYPED_VALUE_INDEX].Values<uint32_t>();
		for (idx_t row = 0; row < count; row++) {
			auto untyped_index = untyped_indexes[row];
			if (!node_validity.IsValid(row) || !untyped_index.IsValid() || untyped_index.GetValue() == 0) {
				validity.SetInvalid(row);
				continue;
			}
			value_index_sel.set_index(row, untyped_index.GetValue() - 1);
			has_leftover = true;
		}
	}
	if (!has_leftover) {
		ConstantVector::SetNull(result, count_t(count));
		return;
	}
	ExtractFromValues(unshredded, components, path_index, value_index_sel, validity, result, count);
}

void VariantUtils::VariantExtract(const Vector &variant_vec, const vector<VariantPathComponent> &components,
                                  Vector &result, idx_t count) {
	if (TryFromShreddedExtract(variant_vec, components, result, count)) {
		return;
	}
	//! Extract always starts by looking at value_index 0
	SelectionVector value_index_sel(count);
	for (idx_t i = 0; i < count; i++) {
		value_index_sel[i] = 0;
	}
	ValidityMask validity(count);
	ExtractFromValues(variant_vec, components, 0, value_index_sel, validity, result, count);
}

//! FIXME: it could make sense to allow a third argument: 'default'
//! This can currently be achieved with COALESCE(TRY(<extract method>), 'default')
static void VariantExtractFunction(DataChunk &input, ExpressionState &state, Vector &result) {
	auto count = input.size();

	D_ASSERT(input.ColumnCount() == 2);
	const auto &variant_vec = input.data[0];
	D_ASSERT(variant_vec.GetType() == LogicalType::VARIANT());

	const auto &path = input.data[1];
	D_ASSERT(path.GetVectorType() == VectorType::CONSTANT_VECTOR);
	(void)path;

	auto &func_expr = state.expr.Cast<BoundFunctionExpression>();
	auto &info = func_expr.BindInfo()->Cast<VariantExtractBindData>();
	VariantUtils::VariantExtract(variant_vec, {info.component}, result, count);
}

ScalarFunctionSet VariantExtractFun::GetFunctions() {
	auto variant_type = LogicalType::VARIANT();

	ScalarFunctionSet fun_set;
	ScalarFunction variant_extract("variant_extract", {}, variant_type, VariantExtractFunction, VariantExtractBind,
	                               VariantExtractPropagateStats);

	variant_extract.GetSignature().AddParameter("input_variant", variant_type);
	variant_extract.GetSignature().AddParameter("field", LogicalType::VARCHAR);
	fun_set.AddFunction(variant_extract);

	variant_extract.GetSignature().GetParameter(1).SetType(LogicalType::UINTEGER);
	variant_extract.GetSignature().GetParameter(1).SetName("index");
	fun_set.AddFunction(variant_extract);
	return fun_set;
}

} // namespace duckdb
