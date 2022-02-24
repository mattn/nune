// Copyright © The Nune Author. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nune_test

import (
	"testing"
	"time"

	"github.com/vorduin/nune"
)

func benchmarkMilli(b *testing.B, f func()) {
	b.ResetTimer()

	start := time.Now()
	for i := 0; i < b.N; i++ {
		f()
	}
	execTime := time.Since(start)

	b.ReportMetric(0, "ns/op")
	b.ReportMetric((1e3*execTime.Seconds())/float64(b.N), "ms/op")
}

func BenchmarkMin1e8Procs1(b *testing.B) {
	nune.EnvConfig.NumCPU = 1
	tensor := nune.Range[float64](0, 1e8, 1)

	benchmarkMilli(b, func() {
		tensor.Min()
	})
}

func BenchmarkMin1e8ProcsN(b *testing.B) {
	nune.EnvConfig.NumCPU = 0
	tensor := nune.Range[float64](0, 1e8, 1)

	benchmarkMilli(b, func() {
		tensor.Min()
	})
}

func BenchmarkMax1e8Procs1(b *testing.B) {
	nune.EnvConfig.NumCPU = 1
	tensor := nune.Range[float64](0, 1e8, 1)

	benchmarkMilli(b, func() {
		tensor.Max()
	})
}

func BenchmarkMax1e8ProcsN(b *testing.B) {
	nune.EnvConfig.NumCPU = 0
	tensor := nune.Range[float64](0, 1e8, 1)

	benchmarkMilli(b, func() {
		tensor.Max()
	})
}

func BenchmarkMean1e8Procs1(b *testing.B) {
	nune.EnvConfig.NumCPU = 1
	tensor := nune.Range[float64](0, 1e8, 1)

	benchmarkMilli(b, func() {
		tensor.Mean()
	})
}

func BenchmarkMean1e8ProcsN(b *testing.B) {
	nune.EnvConfig.NumCPU = 0
	tensor := nune.Range[float64](0, 1e8, 1)

	benchmarkMilli(b, func() {
		tensor.Mean()
	})
}

func BenchmarkSum1e8Procs1(b *testing.B) {
	nune.EnvConfig.NumCPU = 1
	tensor := nune.Range[float64](0, 1e8, 1)

	benchmarkMilli(b, func() {
		tensor.Sum()
	})
}

func BenchmarkSum1e8ProcsN(b *testing.B) {
	nune.EnvConfig.NumCPU = 0
	tensor := nune.Range[float64](0, 1e8, 1)

	benchmarkMilli(b, func() {
		tensor.Sum()
	})
}

func BenchmarkProd1e8Procs1(b *testing.B) {
	nune.EnvConfig.NumCPU = 1
	tensor := nune.Range[float64](0, 1e8, 1)

	benchmarkMilli(b, func() {
		tensor.Prod()
	})
}

func BenchmarkProd1e8ProcsN(b *testing.B) {
	nune.EnvConfig.NumCPU = 0
	tensor := nune.Range[float64](0, 1e8, 1)

	benchmarkMilli(b, func() {
		tensor.Prod()
	})
}
