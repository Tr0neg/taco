#include "test.h"
#include "test_tensors.h"
#include "taco/tensor.h"

#include <algorithm>
#include <array>
#include <map>
#include <numeric>
#include <random>
#include <vector>

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

TEST(expr, reduction_minimum_2d) {
	Tensor<double> a("a", { 3 }, Format({ Dense }));
	Tensor<double> b("b", { 4, 3 }, Format({ Dense, Dense }));
	b.insert({ 0, 0 }, 5.0);
	b.insert({ 0, 1 }, -4.0);
	b.insert({ 0, 2 }, 8.0);
	b.insert({ 1, 0 }, 2.0);
	b.insert({ 1, 1 }, 3.0);
	b.insert({ 1, 2 }, -1.0);
	b.insert({ 2, 0 }, 7.0);
	b.insert({ 2, 1 }, -6.0);
	b.insert({ 2, 2 }, 4.0);
	b.insert({ 3, 0 }, -2.0);
	b.insert({ 3, 1 }, 10.0);
	b.insert({ 3, 2 }, -5.0);
	b.pack();

	a(j) = minimum(i, b(i, j));
	a.evaluate();

	Tensor<double> expected("expected", { 3 }, Format({ Dense }));
	expected.insert({ 0 }, -2.0);
	expected.insert({ 1 }, -6.0);
	expected.insert({ 2 }, -5.0);
	expected.pack();
	ASSERT_TRUE(equals(expected, a)) << endl << expected << endl << endl << a;
}

TEST(expr, reduction_maximum_2d) {
	Tensor<double> a("a", { 3 }, Format({ Dense }));
	Tensor<double> b("b", { 4, 3 }, Format({ Dense, Dense }));
	b.insert({ 0, 0 }, 5.0);
	b.insert({ 0, 1 }, -4.0);
	b.insert({ 0, 2 }, 8.0);
	b.insert({ 1, 0 }, 2.0);
	b.insert({ 1, 1 }, 3.0);
	b.insert({ 1, 2 }, -1.0);
	b.insert({ 2, 0 }, 7.0);
	b.insert({ 2, 1 }, -6.0);
	b.insert({ 2, 2 }, 4.0);
	b.insert({ 3, 0 }, -2.0);
	b.insert({ 3, 1 }, 10.0);
	b.insert({ 3, 2 }, -5.0);
	b.pack();

	a(j) = maximum(i, b(i, j));
	a.evaluate();

	Tensor<double> expected("expected", { 3 }, Format({ Dense }));
	expected.insert({ 0 }, 7.0);
	expected.insert({ 1 }, 10.0);
	expected.insert({ 2 }, 8.0);
	expected.pack();
	ASSERT_TRUE(equals(expected, a)) << endl << expected << endl << endl << a;
}

TEST(expr, reduction_minimum_3d) {
	Tensor<double> a("a", { 2, 2 }, Format({ Dense, Dense }));
	Tensor<double> b("b", { 4, 2, 2 }, Format({ Dense, Dense, Dense }));
	b.insert({ 0, 0, 0 }, 6.0);
	b.insert({ 0, 0, 1 }, -2.0);
	b.insert({ 0, 1, 0 }, 9.0);
	b.insert({ 0, 1, 1 }, 4.0);
	b.insert({ 1, 0, 0 }, 1.0);
	b.insert({ 1, 0, 1 }, 7.0);
	b.insert({ 1, 1, 0 }, -3.0);
	b.insert({ 1, 1, 1 }, 5.0);
	b.insert({ 2, 0, 0 }, -5.0);
	b.insert({ 2, 0, 1 }, 12.0);
	b.insert({ 2, 1, 0 }, 2.0);
	b.insert({ 2, 1, 1 }, -8.0);
	b.insert({ 3, 0, 0 }, 3.0);
	b.insert({ 3, 0, 1 }, -4.0);
	b.insert({ 3, 1, 0 }, 11.0);
	b.insert({ 3, 1, 1 }, 0.0);
	b.pack();

	a(j, k) = minimum(i, b(i, j, k));
	a.evaluate();

	Tensor<double> expected("expected", { 2, 2 }, Format({ Dense, Dense }));
	expected.insert({ 0, 0 }, -5.0);
	expected.insert({ 0, 1 }, -4.0);
	expected.insert({ 1, 0 }, -3.0);
	expected.insert({ 1, 1 }, -8.0);
	expected.pack();
	ASSERT_TRUE(equals(expected, a)) << endl << expected << endl << endl << a;
}

TEST(expr, reduction_maximum_3d) {
	Tensor<double> a("a", { 2, 2 }, Format({ Dense, Dense }));
	Tensor<double> b("b", { 4, 2, 2 }, Format({ Dense, Dense, Dense }));
	b.insert({ 0, 0, 0 }, 6.0);
	b.insert({ 0, 0, 1 }, -2.0);
	b.insert({ 0, 1, 0 }, 9.0);
	b.insert({ 0, 1, 1 }, 4.0);
	b.insert({ 1, 0, 0 }, 1.0);
	b.insert({ 1, 0, 1 }, 7.0);
	b.insert({ 1, 1, 0 }, -3.0);
	b.insert({ 1, 1, 1 }, 5.0);
	b.insert({ 2, 0, 0 }, -5.0);
	b.insert({ 2, 0, 1 }, 12.0);
	b.insert({ 2, 1, 0 }, 2.0);
	b.insert({ 2, 1, 1 }, -8.0);
	b.insert({ 3, 0, 0 }, 3.0);
	b.insert({ 3, 0, 1 }, -4.0);
	b.insert({ 3, 1, 0 }, 11.0);
	b.insert({ 3, 1, 1 }, 0.0);
	b.pack();

	a(j, k) = maximum(i, b(i, j, k));
	a.evaluate();

	Tensor<double> expected("expected", { 2, 2 }, Format({ Dense, Dense }));
	expected.insert({ 0, 0 }, 6.0);
	expected.insert({ 0, 1 }, 12.0);
	expected.insert({ 1, 0 }, 11.0);
	expected.insert({ 1, 1 }, 5.0);
	expected.pack();
	ASSERT_TRUE(equals(expected, a)) << endl << expected << endl << endl << a;
}

TEST(expr, reduction_minimum_maximum_5d_half_full) {
	constexpr int dimensionSize = 10;
	constexpr int dimensionCount = 5;
	constexpr int totalCoordinates = 100000;
	constexpr int storedCoordinates = totalCoordinates / 2;

	const IndexVar l("l"), m("m");
	const std::vector<int> dimensions(dimensionCount, dimensionSize);
	const std::vector<int> ordering = { 0, 1, 2, 3, 4 };
	const Format inputFormat(
		{ ModeFormatPack(Sparse), ModeFormatPack(Sparse),
		  ModeFormatPack(Sparse), ModeFormatPack(Sparse),
		  ModeFormatPack(Sparse) },
		ordering);
	Tensor<double> b("b", dimensions, inputFormat);

	std::mt19937 generator(20261010);
	std::vector<int> linearCoordinates(totalCoordinates);
	std::iota(linearCoordinates.begin(), linearCoordinates.end(), 0);
	std::shuffle(linearCoordinates.begin(), linearCoordinates.end(), generator);
	std::uniform_int_distribution<int> valueDistribution(-100000, 100000);

	std::map<std::vector<int>, std::pair<double, double>> aggregates[4];
	for (int item = 0; item < storedCoordinates; ++item) {
		int linear = linearCoordinates[item];
		std::array<int, dimensionCount> coordinates;
		for (int mode = dimensionCount - 1; mode >= 0; --mode) {
			coordinates[mode] = linear % dimensionSize;
			linear /= dimensionSize;
		}
		const double value = valueDistribution(generator);
		b.insert({ coordinates[0], coordinates[1], coordinates[2],
		           coordinates[3], coordinates[4] }, value);

		for (int retained = 1; retained <= 4; ++retained) {
			const std::vector<int> key(coordinates.begin(),
			                           coordinates.begin() + retained);
			auto inserted = aggregates[retained - 1].emplace(
				key, std::make_pair(value, value));
			if (!inserted.second) {
				inserted.first->second.first =
					std::min(inserted.first->second.first, value);
				inserted.first->second.second =
					std::max(inserted.first->second.second, value);
			}
		}
	}
	b.pack();

	auto sparseFormat = [](int order) {
		std::vector<ModeFormatPack> packs(order, ModeFormatPack(Sparse));
		std::vector<int> modeOrdering(order);
		std::iota(modeOrdering.begin(), modeOrdering.end(), 0);
		return Format(packs, modeOrdering);
	};
	auto fillExpected = [&](Tensor<double>& expected, int retained,
	                        bool useMinimum) {
		for (const auto& aggregate : aggregates[retained - 1]) {
			expected.insert(aggregate.first,
			                useMinimum ? aggregate.second.first
			                           : aggregate.second.second);
		}
		expected.pack();
	};

	Tensor<double> min1("min1", { 10 }, sparseFormat(1));
	min1(i) = minimum(std::vector<IndexVar>{ j, k, l, m }, b(i, j, k, l, m));
	min1.evaluate();
	Tensor<double> expectedMin1("expectedMin1", { 10 }, sparseFormat(1));
	fillExpected(expectedMin1, 1, true);
	ASSERT_TRUE(equals(expectedMin1, min1)) << endl << expectedMin1 << endl << min1;

	Tensor<double> max1("max1", { 10 }, sparseFormat(1));
	max1(i) = maximum(std::vector<IndexVar>{ j, k, l, m }, b(i, j, k, l, m));
	max1.evaluate();
	Tensor<double> expectedMax1("expectedMax1", { 10 }, sparseFormat(1));
	fillExpected(expectedMax1, 1, false);
	ASSERT_TRUE(equals(expectedMax1, max1)) << endl << expectedMax1 << endl << max1;

	Tensor<double> min2("min2", { 10, 10 }, sparseFormat(2));
	min2(i, j) = minimum(std::vector<IndexVar>{ k, l, m }, b(i, j, k, l, m));
	min2.evaluate();
	Tensor<double> expectedMin2("expectedMin2", { 10, 10 }, sparseFormat(2));
	fillExpected(expectedMin2, 2, true);
	ASSERT_TRUE(equals(expectedMin2, min2)) << endl << expectedMin2 << endl << min2;

	Tensor<double> max2("max2", { 10, 10 }, sparseFormat(2));
	max2(i, j) = maximum(std::vector<IndexVar>{ k, l, m }, b(i, j, k, l, m));
	max2.evaluate();
	Tensor<double> expectedMax2("expectedMax2", { 10, 10 }, sparseFormat(2));
	fillExpected(expectedMax2, 2, false);
	ASSERT_TRUE(equals(expectedMax2, max2)) << endl << expectedMax2 << endl << max2;

	Tensor<double> min3("min3", { 10, 10, 10 }, sparseFormat(3));
	min3(i, j, k) = minimum(std::vector<IndexVar>{ l, m }, b(i, j, k, l, m));
	min3.evaluate();
	Tensor<double> expectedMin3("expectedMin3", { 10, 10, 10 }, sparseFormat(3));
	fillExpected(expectedMin3, 3, true);
	ASSERT_TRUE(equals(expectedMin3, min3)) << endl << expectedMin3 << endl << min3;

	Tensor<double> max3("max3", { 10, 10, 10 }, sparseFormat(3));
	max3(i, j, k) = maximum(std::vector<IndexVar>{ l, m }, b(i, j, k, l, m));
	max3.evaluate();
	Tensor<double> expectedMax3("expectedMax3", { 10, 10, 10 }, sparseFormat(3));
	fillExpected(expectedMax3, 3, false);
	ASSERT_TRUE(equals(expectedMax3, max3)) << endl << expectedMax3 << endl << max3;

	Tensor<double> min4("min4", { 10, 10, 10, 10 }, sparseFormat(4));
	min4(i, j, k, l) = minimum(m, b(i, j, k, l, m));
	min4.evaluate();
	Tensor<double> expectedMin4("expectedMin4", { 10, 10, 10, 10 }, sparseFormat(4));
	fillExpected(expectedMin4, 4, true);
	ASSERT_TRUE(equals(expectedMin4, min4)) << endl << expectedMin4 << endl << min4;

	Tensor<double> max4("max4", { 10, 10, 10, 10 }, sparseFormat(4));
	max4(i, j, k, l) = maximum(m, b(i, j, k, l, m));
	max4.evaluate();
	Tensor<double> expectedMax4("expectedMax4", { 10, 10, 10, 10 }, sparseFormat(4));
	fillExpected(expectedMax4, 4, false);
	ASSERT_TRUE(equals(expectedMax4, max4)) << endl << expectedMax4 << endl << max4;
}

TEST(expr, reduction_average_dense) {
	Tensor<double> a("a");
	Tensor<double> b("b", { 4 }, Format({ Dense }));
	b.insert({ 0 }, 1.0);
	b.insert({ 1 }, 3.0);
	b.insert({ 2 }, 8.0);
	b.insert({ 3 }, 0.0);
	b.pack();

	a = average(i, b(i));
	a.evaluate();

	Tensor<double> expected("expected");
	expected.insert({}, 3.0);
	expected.pack();
	ASSERT_TRUE(equals(expected, a)) << endl << expected << endl << a;
}

TEST(expr, reduction_average_integer_input) {
	Tensor<double> a("a");
	Tensor<int> b("b", { 2 }, Format({ Dense }));
	b.insert({ 0 }, 1);
	b.insert({ 1 }, 2);
	b.pack();

	a = average(i, b(i));
	a.evaluate();

	Tensor<double> expected("expected");
	expected.insert({}, 1.5);
	expected.pack();
	ASSERT_TRUE(equals(expected, a)) << endl << expected << endl << a;
}

TEST(expr, reduction_average_sparse) {
	Tensor<double> a("a");
	Tensor<double> b("b", { 10 }, Format({ Sparse }, { 0 }));
	b.insert({ 1 }, 3.0);
	b.insert({ 4 }, 5.0);
	b.insert({ 8 }, 0.0);
	b.insert({ 9 }, 4.0);
	b.pack();

	a = average(i, b(i));
	a.evaluate();

	Tensor<double> expected("expected");
	expected.insert({}, 3.0);
	expected.pack();
	ASSERT_TRUE(equals(expected, a)) << endl << expected << endl << a;
}

TEST(expr, reduction_average_empty_sparse) {
	Tensor<double> a("a");
	Tensor<double> b("b", { 8 }, Format({ Sparse }, { 0 }));
	b.pack();

	a = average(i, b(i));
	a.evaluate();

	Tensor<double> expected("expected");
	expected.insert({}, 0.0);
	expected.pack();
	ASSERT_TRUE(equals(expected, a)) << endl << expected << endl << a;
}

TEST(expr, reduction_average_multiple_indices) {
	Tensor<double> a("a");
	Tensor<double> b("b", { 2, 3 }, Format({ Sparse, Sparse }));
	b.insert({ 0, 0 }, 1.0);
	b.insert({ 0, 1 }, 3.0);
	b.insert({ 1, 2 }, 9.0);
	b.pack();

	a = average(std::vector<IndexVar>{ i, j }, b(i, j));
	a.evaluate();

	Tensor<double> expected("expected");
	expected.insert({}, 13.0 / 3.0);
	expected.pack();
	ASSERT_TRUE(equals(expected, a)) << endl << expected << endl << a;
}

TEST(expr, reduction_average_per_output_sparse) {
	Tensor<double> a("a", { 2 }, Format({ Dense }));
	Tensor<double> b("b", { 4, 2 }, Format({ Sparse, Sparse }));
	b.insert({ 0, 0 }, 2.0);
	b.insert({ 3, 0 }, 4.0);
	b.insert({ 0, 1 }, 3.0);
	b.insert({ 1, 1 }, 6.0);
	b.insert({ 2, 1 }, 9.0);
	b.pack();

	a(j) = average(i, b(i, j));
	a.evaluate();

	Tensor<double> expected("expected", { 2 }, Format({ Dense }));
	expected.insert({ 0 }, 3.0);
	expected.insert({ 1 }, 6.0);
	expected.pack();
	ASSERT_TRUE(equals(expected, a)) << endl << expected << endl << a;
}

TEST(expr, reduction_average_sparse_output) {
	Tensor<double> a("a", { 2 }, Format({ Sparse }, { 0 }));
	Tensor<double> b("b", { 2, 4 }, Format({ Sparse, Sparse }));
	b.insert({ 0, 0 }, 2.0);
	b.insert({ 0, 3 }, 4.0);
	b.insert({ 1, 0 }, 3.0);
	b.insert({ 1, 1 }, 6.0);
	b.insert({ 1, 2 }, 9.0);
	b.pack();

	a(i) = average(j, b(i, j));
	a.evaluate();

	Tensor<double> expected("expected", { 2 }, Format({ Sparse }, { 0 }));
	expected.insert({ 0 }, 3.0);
	expected.insert({ 1 }, 6.0);
	expected.pack();
	ASSERT_TRUE(equals(expected, a)) << endl << expected << endl << a;
}

TEST(expr, reduction_counts_sparse_including_zero) {
	Tensor<double> a("a");
	Tensor<double> b("b", { 8 }, Format({ Sparse }, { 0 }));
	b.insert({ 1 }, 0.0);
	b.insert({ 4 }, 17.0);
	b.insert({ 7 }, -2.0);
	b.pack();

	a = counts(i, b(i));
	a.evaluate();

	Tensor<double> expected("expected");
	expected.insert({}, 3.0);
	expected.pack();
	ASSERT_TRUE(equals(expected, a)) << endl << expected << endl << a;
}

TEST(expr, reduction_counts_multiple_indices) {
	Tensor<double> a("a");
	Tensor<double> b("b", { 2, 3 }, Format({ Sparse, Sparse }));
	b.insert({ 0, 0 }, 0.0);
	b.insert({ 0, 1 }, 6.0);
	b.insert({ 1, 2 }, 8.0);
	b.pack();

	a = counts(std::vector<IndexVar>{ i, j }, b(i, j));
	a.evaluate();

	Tensor<double> expected("expected");
	expected.insert({}, 3.0);
	expected.pack();
	ASSERT_TRUE(equals(expected, a)) << endl << expected << endl << a;
}

TEST(expr, reduction_counts_per_output_sparse) {
	Tensor<double> a("a", { 2 }, Format({ Dense }));
	Tensor<double> b("b", { 4, 2 }, Format({ Sparse, Sparse }));
	b.insert({ 0, 0 }, 2.0);
	b.insert({ 3, 0 }, 4.0);
	b.insert({ 0, 1 }, 0.0);
	b.insert({ 1, 1 }, 6.0);
	b.insert({ 2, 1 }, 9.0);
	b.pack();

	a(j) = counts(i, b(i, j));
	a.evaluate();

	Tensor<double> expected("expected", { 2 }, Format({ Dense }));
	expected.insert({ 0 }, 2.0);
	expected.insert({ 1 }, 3.0);
	expected.pack();
	ASSERT_TRUE(equals(expected, a)) << endl << expected << endl << a;
}

TEST(expr, reduction_counts_sparse_output) {
	Tensor<double> a("a", { 2 }, Format({ Sparse }, { 0 }));
	Tensor<double> b("b", { 2, 4 }, Format({ Sparse, Sparse }));
	b.insert({ 0, 0 }, 2.0);
	b.insert({ 0, 3 }, 4.0);
	b.insert({ 1, 1 }, 0.0);
	b.insert({ 1, 2 }, 6.0);
	b.insert({ 1, 3 }, 9.0);
	b.pack();

	a(i) = counts(j, b(i, j));
	a.evaluate();

	Tensor<double> expected("expected", { 2 }, Format({ Sparse }, { 0 }));
	expected.insert({ 0 }, 2.0);
	expected.insert({ 1 }, 3.0);
	expected.pack();
	ASSERT_TRUE(equals(expected, a)) << endl << expected << endl << a;
}

TEST(expr, reduction_countsdist_dense) {
	Tensor<double> a("a");
	Tensor<double> b("b", { 7 }, Format({ Dense }));
	b.insert({ 0 }, 0.0);
	b.insert({ 1 }, 2.0);
	b.insert({ 2 }, 2.0);
	b.insert({ 3 }, -1.0);
	b.insert({ 4 }, 0.0);
	b.insert({ 5 }, 5.0);
	b.insert({ 6 }, -1.0);
	b.pack();

	a = countsdist(i, b(i));
	a.evaluate();

	Tensor<double> expected("expected");
	expected.insert({}, 4.0);
	expected.pack();
	ASSERT_TRUE(equals(expected, a)) << endl << expected << endl << a;
}

TEST(expr, reduction_countsdist_multiple_indices) {
	Tensor<double> a("a");
	Tensor<double> b("b", { 2, 3 }, Format({ Sparse, Sparse }));
	b.insert({ 0, 0 }, 1.0);
	b.insert({ 0, 1 }, 1.0);
	b.insert({ 1, 2 }, 9.0);
	b.pack();

	a = countsdist(std::vector<IndexVar>{ i, j }, b(i, j));
	a.evaluate();

	Tensor<double> expected("expected");
	expected.insert({}, 2.0);
	expected.pack();
	ASSERT_TRUE(equals(expected, a)) << endl << expected << endl << a;
}

TEST(expr, reduction_countsdist_sparse_including_zero) {
	Tensor<double> a("a");
	Tensor<double> b("b", { 10 }, Format({ Sparse }, { 0 }));
	b.insert({ 1 }, 7.0);
	b.insert({ 3 }, 7.0);
	b.insert({ 6 }, 0.0);
	b.insert({ 9 }, -2.0);
	b.pack();

	a = countsdist(i, b(i));
	a.evaluate();

	Tensor<double> expected("expected");
	expected.insert({}, 3.0);
	expected.pack();
	ASSERT_TRUE(equals(expected, a)) << endl << expected << endl << a;
}

TEST(expr, reduction_countsdist_per_output) {
	Tensor<double> a("a", { 2 }, Format({ Dense }));
	Tensor<double> b("b", { 4, 2 }, Format({ Dense, Dense }));
	b.insert({ 0, 0 }, 1.0);
	b.insert({ 0, 1 }, -2.0);
	b.insert({ 1, 0 }, 1.0);
	b.insert({ 1, 1 }, -2.0);
	b.insert({ 2, 0 }, 2.0);
	b.insert({ 2, 1 }, -2.0);
	b.insert({ 3, 0 }, 3.0);
	b.insert({ 3, 1 }, 0.0);
	b.pack();

	a(j) = countsdist(i, b(i, j));
	a.evaluate();

	Tensor<double> expected("expected", { 2 }, Format({ Dense }));
	expected.insert({ 0 }, 3.0);
	expected.insert({ 1 }, 2.0);
	expected.pack();
	ASSERT_TRUE(equals(expected, a)) << endl << expected << endl << endl << a;
}

TEST(expr, reduction_countsdist_per_output_sparse) {
	Tensor<double> a("a", { 2 }, Format({ Dense }));
	Tensor<double> b("b", { 4, 2 }, Format({ Sparse, Sparse }));
	b.insert({ 0, 0 }, 1.0);
	b.insert({ 1, 0 }, 1.0);
	b.insert({ 2, 0 }, 2.0);
	b.insert({ 0, 1 }, -2.0);
	b.insert({ 1, 1 }, -2.0);
	b.insert({ 3, 1 }, 0.0);
	b.pack();

	a(j) = countsdist(i, b(i, j));
	a.evaluate();

	Tensor<double> expected("expected", { 2 }, Format({ Dense }));
	expected.insert({ 0 }, 2.0);
	expected.insert({ 1 }, 2.0);
	expected.pack();
	ASSERT_TRUE(equals(expected, a)) << endl << expected << endl << a;
}

TEST(expr, reduction_countsdist_sparse_output) {
	Tensor<double> a("a", { 2 }, Format({ Sparse }, { 0 }));
	Tensor<double> b("b", { 2, 4 }, Format({ Sparse, Sparse }));
	b.insert({ 0, 0 }, 2.0);
	b.insert({ 0, 3 }, 2.0);
	b.insert({ 1, 0 }, 0.0);
	b.insert({ 1, 1 }, 6.0);
	b.insert({ 1, 2 }, 6.0);
	b.pack();

	a(i) = countsdist(j, b(i, j));
	a.evaluate();

	Tensor<double> expected("expected", { 2 }, Format({ Sparse }, { 0 }));
	expected.insert({ 0 }, 1.0);
	expected.insert({ 1 }, 2.0);
	expected.pack();
	ASSERT_TRUE(equals(expected, a)) << endl << expected << endl << a;
}

TEST(expr, reduction_counts_and_countsdist_strings) {
  Tensor<std::string> input("labels", {3, 2}, Format({Sparse, Sparse}));
  input.insert({0, 0}, "яблоко");
  input.insert({1, 0}, "яблоко");
  input.insert({2, 0}, "");
  input.insert({0, 1}, "банан");
  input.insert({2, 1}, "банан");
  input.pack();

  ASSERT_EQ(input.getComponentType(), type<std::string>());
  const std::vector<int> firstLabelCoordinate{0, 0};
  const std::vector<int> emptyLabelCoordinate{2, 0};
  ASSERT_EQ(input.at(firstLabelCoordinate), "яблоко");
  ASSERT_EQ(input.at(emptyLabelCoordinate), "");

  Tensor<long long> countsOutput("label_counts", {2}, Format({Dense}));
  countsOutput(j) = counts(i, input(i, j));
  countsOutput.evaluate();
  const std::vector<int> firstGroupCoordinate{0};
  const std::vector<int> secondGroupCoordinate{1};
  EXPECT_EQ(countsOutput.at(firstGroupCoordinate), 3);
  EXPECT_EQ(countsOutput.at(secondGroupCoordinate), 2);

  Tensor<long long> distinctOutput("label_distinct", {2}, Format({Dense}));
  distinctOutput(j) = countsdist(i, input(i, j));
  distinctOutput.evaluate();
  EXPECT_EQ(distinctOutput.at(firstGroupCoordinate), 2);
  EXPECT_EQ(distinctOutput.at(secondGroupCoordinate), 1);

  size_t iteratedValues = 0;
  for (const auto& value : input) {
    if (value.second.empty()) {
      EXPECT_EQ(value.first[0], 2);
    }
    ++iteratedValues;
  }
  EXPECT_EQ(iteratedValues, 5u);
}

TEST(expr, reduction_counts_and_countsdist_dates) {
  Tensor<Date> input("dates", {5}, Format({Sparse}));
  input.insert({0}, Date(2025, 1, 2));
  input.insert({1}, Date(2024, 12, 31));
  input.insert({2}, Date(2025, 1, 2));
  input.insert({3}, Date(2025, 2, 1));
  input.insert({4}, Date(2024, 12, 31));
  input.pack();

  ASSERT_EQ(input.getComponentType(), type<Date>());
  const std::vector<int> firstCoordinate{0};
  const std::vector<int> scalarCoordinate;
  ASSERT_TRUE(input.at(firstCoordinate) == Date(2025, 1, 2));

  Tensor<long long> countOutput("date_count");
  countOutput = counts(i, input(i));
  countOutput.evaluate();
  EXPECT_EQ(countOutput.at(scalarCoordinate), 5);

  Tensor<long long> distinctOutput("date_distinct");
  distinctOutput = countsdist(i, input(i));
  distinctOutput.evaluate();
  EXPECT_EQ(distinctOutput.at(scalarCoordinate), 3);
}

TEST(expr, reduction_counts_and_countsdist_datetimes) {
  const Date day(2025, 5, 6);
  const DateTime morning(day, 9, 15, 30, 123456);
  Tensor<DateTime> input("timestamps", {4}, Format({Sparse}));
  input.insert({0}, morning);
  input.insert({1}, DateTime(day, 9, 15, 30, 123456));
  input.insert({2}, DateTime(day, 9, 15, 30, 123457));
  input.insert({3}, DateTime(day, 10, 0, 0));
  input.pack();

  ASSERT_EQ(input.getComponentType(), type<DateTime>());
  const std::vector<int> firstCoordinate{0};
  const std::vector<int> scalarCoordinate;
  ASSERT_TRUE(input.at(firstCoordinate) == morning);

  Tensor<long long> countOutput("datetime_count");
  countOutput = counts(i, input(i));
  countOutput.evaluate();
  EXPECT_EQ(countOutput.at(scalarCoordinate), 4);

  Tensor<long long> distinctOutput("datetime_distinct");
  distinctOutput = countsdist(i, input(i));
  distinctOutput.evaluate();
  EXPECT_EQ(distinctOutput.at(scalarCoordinate), 3);
}

TEST(expr, reduction_counts_empty_sparse) {
	Tensor<double> a("a");
	Tensor<double> b("b", { 8 }, Format({ Sparse }, { 0 }));
	b.pack();

	a = counts(i, b(i));
	a.evaluate();

	Tensor<double> expected("expected");
	expected.insert({}, 0.0);
	expected.pack();
	ASSERT_TRUE(equals(expected, a)) << endl << expected << endl << a;
}

TEST(expr, reduction_countsdist_empty_sparse) {
	Tensor<double> a("a");
	Tensor<double> b("b", { 8 }, Format({ Sparse }, { 0 }));
	b.pack();

	a = countsdist(i, b(i));
	a.evaluate();

	Tensor<double> expected("expected");
	expected.insert({}, 0.0);
	expected.pack();
	ASSERT_TRUE(equals(expected, a)) << endl << expected << endl << a;
}
