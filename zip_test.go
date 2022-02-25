// Copyright © The Nune Author. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nune_test

import (
	"testing"
)

func BenchmarkAdd(b *testing.B) {
	tensor := newTensor()

	benchmarkOp(b, func() {
		tensor.Add(tensor)
	})
}
func BenchmarkSub(b *testing.B) {
	tensor := newTensor()

	benchmarkOp(b, func() {
		tensor.Sub(tensor)
	})
}

func BenchmarkMul(b *testing.B) {
	tensor := newTensor()

	benchmarkOp(b, func() {
		tensor.Mul(tensor)
	})
}

func BenchmarkDiv(b *testing.B) {
	tensor := newTensor()

	benchmarkOp(b, func() {
		tensor.Div(tensor)
	})
}