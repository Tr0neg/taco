#include "test.h"
#include "test_tensors.h"
#include "taco/tensor.h"

using namespace taco;

const IndexVar i("i"), j("j"), k("k");

TEST(expr, reduction0) {
  Tensor<double> a("a");

  Tensor<double> b = d5a("b", Dense);
  Tensor<double> c = d5b("c", Dense);
  b.pack();
  c.pack();

  a = sum(i, b(i)*c(i));
  a.evaluate();

  Tensor<double> expected("expected");
  expected.insert({}, 40.0);
  expected.pack();
  ASSERT_TRUE(equals(expected,a)) << endl << expected << endl << endl << a;;
}

TEST(expr, reduction_minimum) {
	Tensor<double> a("a");
	Tensor<double> b("b", { 5 }, Format({ Dense }));
	b.insert({ 0 }, 4.0);
	b.insert({ 1 }, -3.0);
	b.insert({ 2 }, 7.0);
	b.insert({ 3 }, -3.0);
	b.insert({ 4 }, 0.0);
	b.pack();

	a = minimum(i, b(i));
	a.evaluate();

	Tensor<double> expected("expected");
	expected.insert({}, -3.0);
	expected.pack();
	ASSERT_TRUE(equals(expected, a)) << endl << expected << endl << endl << a;
}

TEST(expr, reduction_maximum) {
	Tensor<double> a("a");
	Tensor<double> b("b", { 5 }, Format({ Dense }));
	b.insert({ 0 }, 4.0);
	b.insert({ 1 }, -3.0);
	b.insert({ 2 }, 7.0);
	b.insert({ 3 }, -3.0);
	b.insert({ 4 }, 0.0);
	b.pack();

	a = maximum(i, b(i));
	a.evaluate();

	Tensor<double> expected("expected");
	expected.insert({}, 7.0);
	expected.pack();
	ASSERT_TRUE(equals(expected, a)) << endl << expected << endl << endl << a;
}

TEST(expr, reduction_minimum_positive) {
	Tensor<double> a("a");
	Tensor<double> b("b", { 4 }, Format({ Dense }));
	b.insert({ 0 }, 8.0);
	b.insert({ 1 }, 3.0);
	b.insert({ 2 }, 6.0);
	b.insert({ 3 }, 5.0);
	b.pack();

	a = minimum(i, b(i));
	a.evaluate();

	Tensor<double> expected("expected");
	expected.insert({}, 3.0);
	expected.pack();
	ASSERT_TRUE(equals(expected, a)) << endl << expected << endl << endl << a;
}

TEST(expr, reduction_minimum_negative) {
	Tensor<double> a("a");
	Tensor<double> b("b", { 4 }, Format({ Dense }));
	b.insert({ 0 }, -2.0);
	b.insert({ 1 }, -8.0);
	b.insert({ 2 }, -3.0);
	b.insert({ 3 }, -5.0);
	b.pack();

	a = minimum(i, b(i));
	a.evaluate();

	Tensor<double> expected("expected");
	expected.insert({}, -8.0);
	expected.pack();
	ASSERT_TRUE(equals(expected, a)) << endl << expected << endl << endl << a;
}

TEST(expr, reduction_maximum_positive) {
	Tensor<double> a("a");
	Tensor<double> b("b", { 4 }, Format({ Dense }));
	b.insert({ 0 }, 8.0);
	b.insert({ 1 }, 3.0);
	b.insert({ 2 }, 6.0);
	b.insert({ 3 }, 5.0);
	b.pack();

	a = maximum(i, b(i));
	a.evaluate();

	Tensor<double> expected("expected");
	expected.insert({}, 8.0);
	expected.pack();
	ASSERT_TRUE(equals(expected, a)) << endl << expected << endl << endl << a;
}

TEST(expr, reduction_maximum_negative) {
	Tensor<double> a("a");
	Tensor<double> b("b", { 4 }, Format({ Dense }));
	b.insert({ 0 }, -2.0);
	b.insert({ 1 }, -8.0);
	b.insert({ 2 }, -3.0);
	b.insert({ 3 }, -5.0);
	b.pack();

	a = maximum(i, b(i));
	a.evaluate();

	Tensor<double> expected("expected");
	expected.insert({}, -2.0);
	expected.pack();
	ASSERT_TRUE(equals(expected, a)) << endl << expected << endl << endl << a;
}

TEST(expr, reduction_minimum_duplicate) {
	Tensor<double> a("a");
	Tensor<double> b("b", { 5 }, Format({ Dense }));
	b.insert({ 0 }, 4.0);
	b.insert({ 1 }, -5.0);
	b.insert({ 2 }, 7.0);
	b.insert({ 3 }, -5.0);
	b.insert({ 4 }, 2.0);
	b.pack();

	a = minimum(i, b(i));
	a.evaluate();

	Tensor<double> expected("expected");
	expected.insert({}, -5.0);
	expected.pack();
	ASSERT_TRUE(equals(expected, a)) << endl << expected << endl << endl << a;
}

TEST(expr, reduction_maximum_duplicate) {
	Tensor<double> a("a");
	Tensor<double> b("b", { 5 }, Format({ Dense }));
	b.insert({ 0 }, 8.0);
	b.insert({ 1 }, -5.0);
	b.insert({ 2 }, 3.0);
	b.insert({ 3 }, 8.0);
	b.insert({ 4 }, 2.0);
	b.pack();

	a = maximum(i, b(i));
	a.evaluate();

	Tensor<double> expected("expected");
	expected.insert({}, 8.0);
	expected.pack();
	ASSERT_TRUE(equals(expected, a)) << endl << expected << endl << endl << a;
}

TEST(expr, reduction_minimum_sparse) {
	Tensor<double> a("a");
	Tensor<double> b("b", { 6 }, Format({ Sparse }, { 0 }));
	b.insert({ 0 }, 4.0);
	b.insert({ 2 }, 9.0);
	b.insert({ 5 }, 2.0);
	b.pack();

	a = minimum(i, b(i));
	a.evaluate();

	Tensor<double> expected("expected");
	expected.insert({}, 2.0);
	expected.pack();
	ASSERT_TRUE(equals(expected, a)) << endl << expected << endl << endl << a;
}

TEST(expr, reduction_maximum_sparse) {
	Tensor<double> a("a");
	Tensor<double> b("b", { 6 }, Format({ Sparse }, { 0 }));
	b.insert({ 0 }, 4.0);
	b.insert({ 2 }, 9.0);
	b.insert({ 5 }, 2.0);
	b.pack();

	a = maximum(i, b(i));
	a.evaluate();

	Tensor<double> expected("expected");
	expected.insert({}, 9.0);
	expected.pack();
	ASSERT_TRUE(equals(expected, a)) << endl << expected << endl << endl << a;
}