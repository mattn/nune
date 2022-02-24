// Copyright © The Nune Author. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nune_test

import (
	"testing"

	"github.com/vorduin/nune"
)

func BenchmarkAbs1e8Procs1(b *testing.B) {
	nune.EnvConfig.NumCPU = 1
	tensor := nune.Range[float64](0, 1e8, 1)

	benchmarkMilli(b, func() {
		tensor.Abs()
	})
}

func BenchmarkAbs1e8ProcsN(b *testing.B) {
	nune.EnvConfig.NumCPU = 0
	tensor := nune.Range[float64](0, 1e8, 1)

	benchmarkMilli(b, func() {
		tensor.Abs()
	})
}

func BenchmarkSin1e8Procs1(b *testing.B) {
	nune.EnvConfig.NumCPU = 1
	tensor := nune.Range[float64](0, 1e8, 1)

	benchmarkMilli(b, func() {
		tensor.Sin()
	})
}

func BenchmarkSin1e8ProcsN(b *testing.B) {
	nune.EnvConfig.NumCPU = 0
	tensor := nune.Range[float64](0, 1e8, 1)

	benchmarkMilli(b, func() {
		tensor.Sin()
	})
}

func BenchmarkSqrt1e8Procs1(b *testing.B) {
	nune.EnvConfig.NumCPU = 1
	tensor := nune.Range[float64](0, 1e8, 1)

	benchmarkMilli(b, func() {
		tensor.Sqrt()
	})
}

func BenchmarkSqrt1e8ProcsN(b *testing.B) {
	nune.EnvConfig.NumCPU = 0
	tensor := nune.Range[float64](0, 1e8, 1)

	benchmarkMilli(b, func() {
		tensor.Sqrt()
	})
}

func BenchmarkCbrt1e8Procs1(b *testing.B) {
	nune.EnvConfig.NumCPU = 1
	tensor := nune.Range[float64](0, 1e8, 1)

	benchmarkMilli(b, func() {
		tensor.Cbrt()
	})
}

func BenchmarkCbrt1e8ProcsN(b *testing.B) {
	nune.EnvConfig.NumCPU = 0
	tensor := nune.Range[float64](0, 1e8, 1)

	benchmarkMilli(b, func() {
		tensor.Cbrt()
	})
}

func BenchmarkExp1e8Procs1(b *testing.B) {
	nune.EnvConfig.NumCPU = 1
	tensor := nune.Range[float64](0, 1e8, 1)

	benchmarkMilli(b, func() {
		tensor.Exp()
	})
}

func BenchmarkExp1e8ProcsN(b *testing.B) {
	nune.EnvConfig.NumCPU = 0
	tensor := nune.Range[float64](0, 1e8, 1)

	benchmarkMilli(b, func() {
		tensor.Exp()
	})
}

func BenchmarkLog1e8Procs1(b *testing.B) {
	nune.EnvConfig.NumCPU = 1
	tensor := nune.Range[float64](0, 1e8, 1)

	benchmarkMilli(b, func() {
		tensor.Log()
	})
}

func BenchmarkLog1e8ProcsN(b *testing.B) {
	nune.EnvConfig.NumCPU = 0
	tensor := nune.Range[float64](0, 1e8, 1)

	benchmarkMilli(b, func() {
		tensor.Log()
	})
}

func BenchmarkRound1e8Procs1(b *testing.B) {
	nune.EnvConfig.NumCPU = 1
	tensor := nune.Range[float64](0, 1e8, 1)

	benchmarkMilli(b, func() {
		tensor.Round()
	})
}

func BenchmarkRound1e8ProcsN(b *testing.B) {
	nune.EnvConfig.NumCPU = 0
	tensor := nune.Range[float64](0, 1e8, 1)

	benchmarkMilli(b, func() {
		tensor.Round()
	})
}

func BenchmarkCeil1e8Procs1(b *testing.B) {
	nune.EnvConfig.NumCPU = 1
	tensor := nune.Range[float64](0, 1e8, 1)

	benchmarkMilli(b, func() {
		tensor.Ceil()
	})
}

func BenchmarkCeil1e8ProcsN(b *testing.B) {
	nune.EnvConfig.NumCPU = 0
	tensor := nune.Range[float64](0, 1e8, 1)

	benchmarkMilli(b, func() {
		tensor.Ceil()
	})
}

func BenchmarkTrunc1e8Procs1(b *testing.B) {
	nune.EnvConfig.NumCPU = 1
	tensor := nune.Range[float64](0, 1e8, 1)

	benchmarkMilli(b, func() {
		tensor.Trunc()
	})
}

func BenchmarkTrunc1e8ProcsN(b *testing.B) {
	nune.EnvConfig.NumCPU = 0
	tensor := nune.Range[float64](0, 1e8, 1)

	benchmarkMilli(b, func() {
		tensor.Trunc()
	})
}
