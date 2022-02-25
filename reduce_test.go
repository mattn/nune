// Copyright © The Nune Author. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nune_test

import (
	"testing"
)

func BenchmarkMin(b *testing.B) {
	tensor := newTensor()

	benchmarkOp(b, func() {
		tensor.Min()
	})
}

func BenchmarkMax(b *testing.B) {
	tensor := newTensor()

	benchmarkOp(b, func() {
		tensor.Max()
	})
}

func BenchmarkMean(b *testing.B) {
	tensor := newTensor()

	benchmarkOp(b, func() {
		tensor.Mean()
	})
}

func BenchmarkSum(b *testing.B) {
	tensor := newTensor()

	benchmarkOp(b, func() {
		tensor.Sum()
	})
}

func BenchmarkProd(b *testing.B) {
	tensor := newTensor()

	benchmarkOp(b, func() {
		tensor.Prod()
	})
}