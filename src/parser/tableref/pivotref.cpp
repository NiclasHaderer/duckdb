#include "duckdb/parser/tableref/pivotref.hpp"
#include "duckdb/parser/expression_util.hpp"
#include "duckdb/common/limits.hpp"

#include "duckdb/common/exception/conversion_exception.hpp"
#include "duckdb/parser/expression/between_expression.hpp"
#include "duckdb/parser/expression/case_expression.hpp"
#include "duckdb/parser/expression/cast_expression.hpp"
#include "duckdb/parser/expression/collate_expression.hpp"
#include "duckdb/parser/expression/columnref_expression.hpp"
#include "duckdb/parser/expression/comparison_expression.hpp"
#include "duckdb/parser/expression/conjunction_expression.hpp"
#include "duckdb/parser/expression/constant_expression.hpp"
#include "duckdb/parser/expression/function_expression.hpp"
#include "duckdb/parser/expression/operator_expression.hpp"
#include "duckdb/parser/expression/star_expression.hpp"
#include "duckdb/common/serializer/serializer.hpp"
#include "duckdb/common/serializer/deserializer.hpp"
#include "duckdb/common/types/value.hpp"

namespace duckdb {

//===--------------------------------------------------------------------===//
// PivotColumn
//===--------------------------------------------------------------------===//
string PivotColumn::ToString() const {
	string result;
	if (!unpivot_names.empty()) {
		D_ASSERT(pivot_expressions.empty());
		// unpivot
		if (unpivot_names.size() == 1) {
			result += KeywordHelper::WriteOptionallyQuoted(unpivot_names[0]);
		} else {
			result += "(";
			for (idx_t n = 0; n < unpivot_names.size(); n++) {
				if (n > 0) {
					result += ", ";
				}
				result += KeywordHelper::WriteOptionallyQuoted(unpivot_names[n]);
			}
			result += ")";
		}
	} else if (!pivot_expressions.empty()) {
		// pivot
		result += "(";
		for (idx_t n = 0; n < pivot_expressions.size(); n++) {
			if (n > 0) {
				result += ", ";
			}
			result += pivot_expressions[n]->ToString();
		}
		result += ")";
	}
	result += " IN ";
	if (pivot_enum.empty()) {
		result += "(";
		for (idx_t e = 0; e < entries.size(); e++) {
			auto &entry = entries[e];
			if (e > 0) {
				result += ", ";
			}
			if (entry.expr) {
				D_ASSERT(entry.values.empty());
				result += entry.expr->ToString();
			} else if (entry.values.size() == 1) {
				result += entry.values[0].ToSQLString();
			} else {
				result += "(";
				for (idx_t v = 0; v < entry.values.size(); v++) {
					if (v > 0) {
						result += ", ";
					}
					result += entry.values[v].ToSQLString();
				}
				result += ")";
			}
			if (!entry.alias.empty()) {
				result += " AS " + KeywordHelper::WriteOptionallyQuoted(entry.alias);
			}
		}
		result += ")";
	} else {
		result += KeywordHelper::WriteOptionallyQuoted(pivot_enum);
	}
	return result;
}

bool PivotColumnEntry::Equals(const PivotColumnEntry &other) const {
	if (alias != other.alias) {
		return false;
	}
	if (values.size() != other.values.size()) {
		return false;
	}
	for (idx_t i = 0; i < values.size(); i++) {
		if (!Value::NotDistinctFrom(values[i], other.values[i])) {
			return false;
		}
	}
	return true;
}

bool PivotColumn::Equals(const PivotColumn &other) const {
	if (!ExpressionUtil::ListEquals(pivot_expressions, other.pivot_expressions)) {
		return false;
	}
	if (other.unpivot_names != unpivot_names) {
		return false;
	}
	if (other.pivot_enum != pivot_enum) {
		return false;
	}
	if (other.entries.size() != entries.size()) {
		return false;
	}
	for (idx_t i = 0; i < entries.size(); i++) {
		if (!entries[i].Equals(other.entries[i])) {
			return false;
		}
	}
	return true;
}

PivotColumn PivotColumn::Copy() const {
	PivotColumn result;
	for (auto &expr : pivot_expressions) {
		result.pivot_expressions.push_back(expr->Copy());
	}
	result.unpivot_names = unpivot_names;
	for (auto &entry : entries) {
		result.entries.push_back(entry.Copy());
	}
	result.pivot_enum = pivot_enum;
	return result;
}

//===--------------------------------------------------------------------===//
// PivotColumnEntry
//===--------------------------------------------------------------------===//
PivotColumnEntry PivotColumnEntry::Copy() const {
	PivotColumnEntry result;
	result.values = values;
	result.expr = expr ? expr->Copy() : nullptr;
	result.alias = alias;
	return result;
}

static bool TryFoldConstantForBackwardsCompatability(const ParsedExpression &expr, Value &value) {
	switch (expr.GetExpressionType()) {
	case ExpressionType::VALUE_CONSTANT: {
		auto &constant = expr.Cast<ConstantExpression>();
		value = constant.value;
		return true;
	}
	case ExpressionType::FUNCTION: {
		auto &function = expr.Cast<FunctionExpression>();
		if (function.function_name == "struct_pack") {
			unordered_set<string> unique_names;
			child_list_t<Value> values;
			values.reserve(function.children.size());
			for (const auto &child : function.children) {
				if (!unique_names.insert(child->GetAlias()).second) {
					return false;
				}
				Value child_value;
				if (!TryFoldConstantForBackwardsCompatability(*child, child_value)) {
					return false;
				}
				values.emplace_back(child->GetAlias(), std::move(child_value));
			}
			value = Value::STRUCT(std::move(values));
			return true;
		} else if (function.function_name == "list_value") {
			vector<Value> values;
			values.reserve(function.children.size());
			for (const auto &child : function.children) {
				Value child_value;
				if (!TryFoldConstantForBackwardsCompatability(*child, child_value)) {
					return false;
				}
				values.emplace_back(std::move(child_value));
			}
			LogicalType child_type(LogicalTypeId::SQLNULL);
			for (auto &child_value : values) {
				child_type = LogicalType::ForceMaxLogicalType(child_type, child_value.type());
			}
			value = Value::LIST(child_type, values);
			return true;
		} else if (function.function_name == "map") {
			Value keys;
			if (!TryFoldConstantForBackwardsCompatability(*function.children[0], keys)) {
				return false;
			}
			Value values;
			if (!TryFoldConstantForBackwardsCompatability(*function.children[1], values)) {
				return false;
			}
			vector<Value> keys_unpacked = ListValue::GetChildren(keys);
			vector<Value> values_unpacked = ListValue::GetChildren(values);
			value = Value::MAP(ListType::GetChildType(keys.type()), ListType::GetChildType(values.type()),
			                   keys_unpacked, values_unpacked);
			return true;
		} else if (function.function_name == "lower" || function.function_name == "upper") {
			if (function.children.size() != 1) {
				return false;
			}
			Value child_value;
			if (!TryFoldConstantForBackwardsCompatability(*function.children[0], child_value)) {
				return false;
			}
			if (child_value.IsNull()) {
				value = Value(LogicalType::VARCHAR);
				return true;
			}
			string error;
			if (!child_value.DefaultTryCastAs(LogicalType::VARCHAR, child_value, &error)) {
				return false;
			}
			auto str = StringValue::Get(child_value);
			if (function.function_name == "lower") {
				value = Value(StringUtil::Lower(str));
			} else {
				value = Value(StringUtil::Upper(str));
			}
			return true;
		} else {
			return false;
		}
	}
	case ExpressionType::OPERATOR_CAST: {
		auto &cast = expr.Cast<CastExpression>();
		Value dummy_value;
		if (!TryFoldConstantForBackwardsCompatability(*cast.child, dummy_value)) {
			return false;
		}
		LogicalType cast_type;
		try {
			cast_type = UnboundType::TryDefaultBind(cast.cast_type);
		} catch (...) {
			return false;
		}
		if (cast_type == LogicalType::INVALID || cast_type == LogicalTypeId::UNBOUND) {
			return false;
		}
		string error_message;
		if (!dummy_value.DefaultTryCastAs(cast_type, value, &error_message)) {
			return false;
		}
		return true;
	}
	case ExpressionType::CASE_EXPR: {
		auto &case_expr = expr.Cast<CaseExpression>();
		for (auto &check : case_expr.case_checks) {
			Value when_val;
			if (!TryFoldConstantForBackwardsCompatability(*check.when_expr, when_val)) {
				return false;
			}
			if (when_val.IsNull()) {
				continue;
			}
			string error;
			if (!when_val.DefaultTryCastAs(LogicalType::BOOLEAN, when_val, &error)) {
				return false;
			}
			if (BooleanValue::Get(when_val)) {
				return TryFoldConstantForBackwardsCompatability(*check.then_expr, value);
			}
		}
		if (case_expr.else_expr) {
			return TryFoldConstantForBackwardsCompatability(*case_expr.else_expr, value);
		}
		value = Value();
		return true;
	}
	case ExpressionType::COMPARE_EQUAL:
	case ExpressionType::COMPARE_NOTEQUAL:
	case ExpressionType::COMPARE_LESSTHAN:
	case ExpressionType::COMPARE_GREATERTHAN:
	case ExpressionType::COMPARE_LESSTHANOREQUALTO:
	case ExpressionType::COMPARE_GREATERTHANOREQUALTO: {
		auto &comp = expr.Cast<ComparisonExpression>();
		Value left_val, right_val;
		if (!TryFoldConstantForBackwardsCompatability(*comp.left, left_val) ||
		    !TryFoldConstantForBackwardsCompatability(*comp.right, right_val)) {
			return false;
		}
		if (left_val.IsNull() || right_val.IsNull()) {
			value = Value(LogicalType::BOOLEAN);
			return true;
		}
		auto type = expr.GetExpressionType();
		if (type == ExpressionType::COMPARE_EQUAL) {
			value = Value::BOOLEAN(left_val == right_val);
		} else if (type == ExpressionType::COMPARE_NOTEQUAL) {
			value = Value::BOOLEAN(left_val != right_val);
		} else if (type == ExpressionType::COMPARE_LESSTHAN) {
			value = Value::BOOLEAN(left_val < right_val);
		} else if (type == ExpressionType::COMPARE_GREATERTHAN) {
			value = Value::BOOLEAN(left_val > right_val);
		} else if (type == ExpressionType::COMPARE_LESSTHANOREQUALTO) {
			value = Value::BOOLEAN(left_val <= right_val);
		} else {
			value = Value::BOOLEAN(left_val >= right_val);
		}
		return true;
	}
	case ExpressionType::CONJUNCTION_AND:
	case ExpressionType::CONJUNCTION_OR: {
		auto &conj = expr.Cast<ConjunctionExpression>();
		bool is_and = (expr.GetExpressionType() == ExpressionType::CONJUNCTION_AND);
		for (auto &child : conj.children) {
			Value child_val;
			if (!TryFoldConstantForBackwardsCompatability(*child, child_val)) {
				return false;
			}
			if (child_val.IsNull()) {
				value = Value(LogicalType::BOOLEAN);
				return true;
			}
			string error;
			if (!child_val.DefaultTryCastAs(LogicalType::BOOLEAN, child_val, &error)) {
				return false;
			}
			bool child_bool = BooleanValue::Get(child_val);
			if (is_and && !child_bool) {
				value = Value::BOOLEAN(false);
				return true;
			}
			if (!is_and && child_bool) {
				value = Value::BOOLEAN(true);
				return true;
			}
		}
		value = Value::BOOLEAN(is_and);
		return true;
	}
	case ExpressionType::COMPARE_BETWEEN: {
		auto &between = expr.Cast<BetweenExpression>();
		Value input_val, lower_val, upper_val;
		if (!TryFoldConstantForBackwardsCompatability(*between.input, input_val) ||
		    !TryFoldConstantForBackwardsCompatability(*between.lower, lower_val) ||
		    !TryFoldConstantForBackwardsCompatability(*between.upper, upper_val)) {
			return false;
		}
		if (input_val.IsNull() || lower_val.IsNull() || upper_val.IsNull()) {
			value = Value(LogicalType::BOOLEAN);
			return true;
		}
		value = Value::BOOLEAN(input_val >= lower_val && input_val <= upper_val);
		return true;
	}
	case ExpressionType::COLLATE: {
		auto &collate = expr.Cast<CollateExpression>();
		return TryFoldConstantForBackwardsCompatability(*collate.child, value);
	}
	case ExpressionType::OPERATOR_COALESCE: {
		auto &op = expr.Cast<OperatorExpression>();
		for (auto &child : op.children) {
			Value child_val;
			if (!TryFoldConstantForBackwardsCompatability(*child, child_val)) {
				return false;
			}
			if (!child_val.IsNull()) {
				value = std::move(child_val);
				return true;
			}
		}
		value = Value();
		return true;
	}
	case ExpressionType::OPERATOR_NOT: {
		auto &op = expr.Cast<OperatorExpression>();
		Value child_val;
		if (!TryFoldConstantForBackwardsCompatability(*op.children[0], child_val)) {
			return false;
		}
		if (child_val.IsNull()) {
			value = Value(LogicalType::BOOLEAN);
			return true;
		}
		string error;
		if (!child_val.DefaultTryCastAs(LogicalType::BOOLEAN, child_val, &error)) {
			return false;
		}
		value = Value::BOOLEAN(!BooleanValue::Get(child_val));
		return true;
	}
	case ExpressionType::OPERATOR_IS_NULL: {
		auto &op = expr.Cast<OperatorExpression>();
		Value child_val;
		if (!TryFoldConstantForBackwardsCompatability(*op.children[0], child_val)) {
			return false;
		}
		value = Value::BOOLEAN(child_val.IsNull());
		return true;
	}
	case ExpressionType::OPERATOR_IS_NOT_NULL: {
		auto &op = expr.Cast<OperatorExpression>();
		Value child_val;
		if (!TryFoldConstantForBackwardsCompatability(*op.children[0], child_val)) {
			return false;
		}
		value = Value::BOOLEAN(!child_val.IsNull());
		return true;
	}
	case ExpressionType::COMPARE_IN: {
		auto &op = expr.Cast<OperatorExpression>();
		Value needle;
		if (!TryFoldConstantForBackwardsCompatability(*op.children[0], needle)) {
			return false;
		}
		if (needle.IsNull()) {
			value = Value(LogicalType::BOOLEAN);
			return true;
		}
		for (idx_t i = 1; i < op.children.size(); i++) {
			Value candidate;
			if (!TryFoldConstantForBackwardsCompatability(*op.children[i], candidate)) {
				return false;
			}
			if (!candidate.IsNull() && needle == candidate) {
				value = Value::BOOLEAN(true);
				return true;
			}
		}
		value = Value::BOOLEAN(false);
		return true;
	}
	case ExpressionType::COMPARE_NOT_IN: {
		auto &op = expr.Cast<OperatorExpression>();
		Value needle;
		if (!TryFoldConstantForBackwardsCompatability(*op.children[0], needle)) {
			return false;
		}
		if (needle.IsNull()) {
			value = Value(LogicalType::BOOLEAN);
			return true;
		}
		for (idx_t i = 1; i < op.children.size(); i++) {
			Value candidate;
			if (!TryFoldConstantForBackwardsCompatability(*op.children[i], candidate)) {
				return false;
			}
			if (!candidate.IsNull() && needle == candidate) {
				value = Value::BOOLEAN(false);
				return true;
			}
		}
		value = Value::BOOLEAN(true);
		return true;
	}
	case ExpressionType::ARRAY_EXTRACT: {
		auto &op = expr.Cast<OperatorExpression>();
		Value source_val, index_val;
		if (!TryFoldConstantForBackwardsCompatability(*op.children[0], source_val) ||
		    !TryFoldConstantForBackwardsCompatability(*op.children[1], index_val)) {
			return false;
		}
		if (source_val.IsNull() || index_val.IsNull()) {
			value = Value();
			return true;
		}
		if (source_val.type().id() == LogicalTypeId::STRUCT) {
			// Struct access by key name
			string error;
			if (!index_val.DefaultTryCastAs(LogicalType::VARCHAR, index_val, &error)) {
				return false;
			}
			auto key = StringValue::Get(index_val);
			auto &children = StructValue::GetChildren(source_val);
			auto child_count = StructType::GetChildCount(source_val.type());
			for (idx_t i = 0; i < child_count; i++) {
				if (StringUtil::CIEquals(StructType::GetChildName(source_val.type(), i), key)) {
					value = children[i];
					return true;
				}
			}
			return false;
		}
		// List/array access by integer index
		string error;
		if (!index_val.DefaultTryCastAs(LogicalType::INTEGER, index_val, &error)) {
			return false;
		}
		auto idx = IntegerValue::Get(index_val);
		auto &children = ListValue::GetChildren(source_val);
		if (idx < 1 || idx > static_cast<int32_t>(children.size())) {
			value = Value();
			return true;
		}
		value = children[idx - 1];
		return true;
	}
	case ExpressionType::STRUCT_EXTRACT: {
		auto &op = expr.Cast<OperatorExpression>();
		Value source_val;
		if (!TryFoldConstantForBackwardsCompatability(*op.children[0], source_val)) {
			return false;
		}
		if (source_val.IsNull()) {
			value = Value();
			return true;
		}
		// child[1] is a string constant with the field name
		Value key_val;
		if (!TryFoldConstantForBackwardsCompatability(*op.children[1], key_val)) {
			return false;
		}
		auto key = StringValue::Get(key_val);
		auto &children = StructValue::GetChildren(source_val);
		auto child_count = StructType::GetChildCount(source_val.type());
		for (idx_t i = 0; i < child_count; i++) {
			if (StringUtil::CIEquals(StructType::GetChildName(source_val.type(), i), key)) {
				value = children[i];
				return true;
			}
		}
		return false;
	}
	case ExpressionType::ARRAY_SLICE: {
		auto &op = expr.Cast<OperatorExpression>();
		Value source_val, start_val, end_val;
		if (!TryFoldConstantForBackwardsCompatability(*op.children[0], source_val)) {
			return false;
		}
		if (source_val.IsNull()) {
			value = Value();
			return true;
		}
		if (!TryFoldConstantForBackwardsCompatability(*op.children[1], start_val) ||
		    !TryFoldConstantForBackwardsCompatability(*op.children[2], end_val)) {
			return false;
		}
		// Only handle string slicing
		if (source_val.type().id() != LogicalTypeId::VARCHAR) {
			return false;
		}
		string error;
		if (!start_val.DefaultTryCastAs(LogicalType::INTEGER, start_val, &error) ||
		    !end_val.DefaultTryCastAs(LogicalType::INTEGER, end_val, &error)) {
			return false;
		}
		auto str = StringValue::Get(source_val);
		auto start = IntegerValue::Get(start_val);
		auto end = IntegerValue::Get(end_val);
		// DuckDB slice is 1-indexed, inclusive on both ends
		if (start < 1) {
			start = 1;
		}
		if (end > static_cast<int32_t>(str.size())) {
			end = static_cast<int32_t>(str.size());
		}
		if (start > end) {
			value = Value("");
		} else {
			value = Value(str.substr(start - 1, end - start + 1));
		}
		return true;
	}
	default:
		return false;
	}
}

static bool TryFoldForBackwardsCompatibility(const unique_ptr<ParsedExpression> &expr, vector<Value> &values) {
	if (!expr) {
		return true;
	}

	switch (expr->GetExpressionType()) {
	case ExpressionType::COLUMN_REF: {
		auto &colref = expr->Cast<ColumnRefExpression>();
		if (colref.IsQualified()) {
			return false;
		}
		values.emplace_back(colref.GetColumnName());
		return true;
	}
	case ExpressionType::FUNCTION: {
		auto &function = expr->Cast<FunctionExpression>();
		if (function.function_name != "row") {
			return false;
		}
		for (auto &child : function.children) {
			if (!TryFoldForBackwardsCompatibility(child, values)) {
				return false;
			}
		}
		return true;
	}
	default: {
		Value val;
		if (!TryFoldConstantForBackwardsCompatability(*expr, val)) {
			return false;
		}
		values.push_back(std::move(val));
		return true;
	}
	}
}

void PivotColumnEntry::Serialize(Serializer &serializer) const {
	if (serializer.ShouldSerialize(7) || !expr) {
		serializer.WritePropertyWithDefault<vector<Value>>(100, "values", values);
		serializer.WritePropertyWithDefault<unique_ptr<ParsedExpression>>(101, "star_expr", expr);
		serializer.WritePropertyWithDefault<string>(102, "alias", alias);
	} else if (expr->GetExpressionType() == ExpressionType::STAR) {
		// StarExpression: write directly to field 101 (star_expr). Old DuckDB versions
		// natively support star_expr in this field, so this is correct backward compat.
		serializer.WritePropertyWithDefault<vector<Value>>(100, "values", values);
		serializer.WritePropertyWithDefault<unique_ptr<ParsedExpression>>(101, "star_expr", expr);
		serializer.WritePropertyWithDefault<string>(102, "alias", alias);
	} else {
		// We used to only support constant values in pivot entries, and folded expressions in the
		// transformer. So we need to serialize in a backwards compatible way here by trying to fold
		// the expression back to constant values.
		vector<Value> dummy_values;
		if (!TryFoldForBackwardsCompatibility(expr, dummy_values)) {
			throw SerializationException(
			    "Cannot serialize arbitrary expression pivot entries when targeting database storage version '%s'",
			    serializer.GetOptions().serialization_compatibility.duckdb_version);
		}
		serializer.WritePropertyWithDefault<vector<Value>>(100, "values", dummy_values);
		serializer.WritePropertyWithDefault<unique_ptr<ParsedExpression>>(101, "star_expr", nullptr);
		serializer.WritePropertyWithDefault<string>(102, "alias", alias);
	}
}

PivotColumnEntry PivotColumnEntry::Deserialize(Deserializer &deserializer) {
	PivotColumnEntry result;
	deserializer.ReadPropertyWithDefault<vector<Value>>(100, "values", result.values);
	deserializer.ReadPropertyWithDefault<unique_ptr<ParsedExpression>>(101, "star_expr", result.expr);
	deserializer.ReadPropertyWithDefault<string>(102, "alias", result.alias);
	return result;
}

//===--------------------------------------------------------------------===//
// PivotRef
//===--------------------------------------------------------------------===//
string PivotRef::ToString() const {
	string result;
	result = source->ToString();
	if (!aggregates.empty()) {
		// pivot
		result += " PIVOT (";
		for (idx_t aggr_idx = 0; aggr_idx < aggregates.size(); aggr_idx++) {
			if (aggr_idx > 0) {
				result += ", ";
			}
			result += aggregates[aggr_idx]->ToString();
			if (!aggregates[aggr_idx]->GetAlias().empty()) {
				result += " AS " + KeywordHelper::WriteOptionallyQuoted(aggregates[aggr_idx]->GetAlias());
			}
		}
	} else {
		// unpivot
		result += " UNPIVOT ";
		if (include_nulls) {
			result += "INCLUDE NULLS ";
		}
		result += "(";
		if (unpivot_names.size() == 1) {
			result += KeywordHelper::WriteOptionallyQuoted(unpivot_names[0]);
		} else {
			result += "(";
			for (idx_t n = 0; n < unpivot_names.size(); n++) {
				if (n > 0) {
					result += ", ";
				}
				result += KeywordHelper::WriteOptionallyQuoted(unpivot_names[n]);
			}
			result += ")";
		}
	}
	result += " FOR";
	for (auto &pivot : pivots) {
		result += " ";
		result += pivot.ToString();
	}
	if (!groups.empty()) {
		result += " GROUP BY ";
		for (idx_t i = 0; i < groups.size(); i++) {
			if (i > 0) {
				result += ", ";
			}
			result += groups[i];
		}
	}
	result += ")";
	if (!alias.empty()) {
		result += " AS " + KeywordHelper::WriteOptionallyQuoted(alias);
		if (!column_name_alias.empty()) {
			result += "(";
			for (idx_t i = 0; i < column_name_alias.size(); i++) {
				if (i > 0) {
					result += ", ";
				}
				result += KeywordHelper::WriteOptionallyQuoted(column_name_alias[i]);
			}
			result += ")";
		}
	}
	return result;
}

bool PivotRef::Equals(const TableRef &other_p) const {
	if (!TableRef::Equals(other_p)) {
		return false;
	}
	auto &other = other_p.Cast<PivotRef>();
	if (!source->Equals(*other.source)) {
		return false;
	}
	if (!ParsedExpression::ListEquals(aggregates, other.aggregates)) {
		return false;
	}
	if (pivots.size() != other.pivots.size()) {
		return false;
	}
	for (idx_t i = 0; i < pivots.size(); i++) {
		if (!pivots[i].Equals(other.pivots[i])) {
			return false;
		}
	}
	if (unpivot_names != other.unpivot_names) {
		return false;
	}
	if (alias != other.alias) {
		return false;
	}
	if (groups != other.groups) {
		return false;
	}
	if (include_nulls != other.include_nulls) {
		return false;
	}
	return true;
}

unique_ptr<TableRef> PivotRef::Copy() {
	auto copy = make_uniq<PivotRef>();
	copy->source = source->Copy();
	for (auto &aggr : aggregates) {
		copy->aggregates.push_back(aggr->Copy());
	}
	copy->unpivot_names = unpivot_names;
	for (auto &entry : pivots) {
		copy->pivots.push_back(entry.Copy());
	}
	copy->groups = groups;
	copy->column_name_alias = column_name_alias;
	copy->include_nulls = include_nulls;
	copy->alias = alias;
	return std::move(copy);
}

} // namespace duckdb
