#include "catch.hpp"
#include "test_helpers.hpp"

#include <thread>

using namespace duckdb;
using namespace std;

#ifdef DUCKDB_DEBUG_STACKTRACE

TEST_CASE("Test stack trace can be resolved", "[stack_trace]") {
	DuckDB db(nullptr);
	Connection con(db);
	auto result = con.SendQuery("this will not work");
	REQUIRE_FAIL(result);

	auto error_data = result->GetErrorObject();
	auto stack_trace = error_data.GetStackTrace();
	REQUIRE(!stack_trace.empty());
}

#endif
