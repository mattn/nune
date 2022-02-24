// Copyright © The Nune Author. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nune_test

import (
	"testing"

	"github.com/vorduin/nune"
)

func BenchmarkAdd1e8Procs1(b *testing.B) {
	nune.EnvConfig.NumCPU = 1
	tensor := nune.Range[float64](0, 1e8, 1)

	benchmarkMilli(b, func() {
		tensor.Add(tensor)
	})
}

func BenchmarkAdd1e8ProcsN(b *testing.B) {
	nune.EnvConfig.NumCPU = 0
	tensor := nune.Range[float64](0, 1e8, 1)

	benchmarkMilli(b, func() {
		tensor.Add(tensor)
	})
}

func BenchmarkSub1e8Procs1(b *testing.B) {
	nune.EnvConfig.NumCPU = 1
	tensor := nune.Range[float64](0, 1e8, 1)

	benchmarkMilli(b, func() {
		tensor.Sub(tensor)
	})
}

func BenchmarkSub1e8ProcsN(b *testing.B) {
	nune.EnvConfig.NumCPU = 0
	tensor := nune.Range[float64](0, 1e8, 1)

	benchmarkMilli(b, func() {
		tensor.Sub(tensor)
	})
}

func BenchmarkMul1e8Procs1(b *testing.B) {
	nune.EnvConfig.NumCPU = 1
	tensor := nune.Range[float64](0, 1e8, 1)

	benchmarkMilli(b, func() {
		tensor.Mul(tensor)
	})
}

func BenchmarkMul1e8ProcsN(b *testing.B) {
	nune.EnvConfig.NumCPU = 0
	tensor := nune.Range[float64](0, 1e8, 1)

	benchmarkMilli(b, func() {
		tensor.Mul(tensor)
	})
}

func BenchmarkDiv1e8Procs1(b *testing.B) {
	nune.EnvConfig.NumCPU = 1
	tensor := nune.Range[float64](0, 1e8, 1)

	benchmarkMilli(b, func() {
		tensor.Div(tensor)
	})
}

func BenchmarkDiv1e8ProcsN(b *testing.B) {
	nune.EnvConfig.NumCPU = 0
	tensor := nune.Range[float64](0, 1e8, 1)

	benchmarkMilli(b, func() {
		tensor.Div(tensor)
	})
}
