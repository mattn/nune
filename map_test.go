// Copyright © The Nune Author. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nune_test

import (
	"testing"
)

func BenchmarkAbs(b *testing.B) {
	tensor := newTensor()

	benchmarkOp(b, func() {
		tensor.Abs()
	})
}

func BenchmarkAcos(b *testing.B) {
	tensor := newTensor()

	benchmarkOp(b, func() {
		tensor.Acos()
	})
}

func BenchmarkAcosh(b *testing.B) {
	tensor := newTensor()

	benchmarkOp(b, func() {
		tensor.Acosh()
	})
}

func BenchmarkAsin(b *testing.B) {
	tensor := newTensor()

	benchmarkOp(b, func() {
		tensor.Asin()
	})
}

func BenchmarkAsinh(b *testing.B) {
	tensor := newTensor()

	benchmarkOp(b, func() {
		tensor.Asinh()
	})
}

func BenchmarkAtan(b *testing.B) {
	tensor := newTensor()

	benchmarkOp(b, func() {
		tensor.Atan()
	})
}

func BenchmarkAtan2(b *testing.B) {
	tensor := newTensor()

	benchmarkOp(b, func() {
		tensor.Atan2(0)
	})
}

func BenchmarkAtanh(b *testing.B) {
	tensor := newTensor()

	benchmarkOp(b, func() {
		tensor.Atanh()
	})
}

func BenchmarkCbrt(b *testing.B) {
	tensor := newTensor()

	benchmarkOp(b, func() {
		tensor.Cbrt()
	})
}

func BenchmarkCeil(b *testing.B) {
	tensor := newTensor()

	benchmarkOp(b, func() {
		tensor.Ceil()
	})
}

func BenchmarkCopysign(b *testing.B) {
	tensor := newTensor()

	benchmarkOp(b, func() {
		tensor.Copysign(-1)
	})
}

func BenchmarkCos(b *testing.B) {
	tensor := newTensor()

	benchmarkOp(b, func() {
		tensor.Cos()
	})
}

func BenchmarkCosh(b *testing.B) {
	tensor := newTensor()

	benchmarkOp(b, func() {
		tensor.Cosh()
	})
}

func BenchmarkDim(b *testing.B) {
	tensor := newTensor()

	benchmarkOp(b, func() {
		tensor.Dim(0)
	})
}

func BenchmarkErf(b *testing.B) {
	tensor := newTensor()

	benchmarkOp(b, func() {
		tensor.Erf()
	})
}

func BenchmarkErfc(b *testing.B) {
	tensor := newTensor()

	benchmarkOp(b, func() {
		tensor.Erfc()
	})
}

func BenchmarkErfcinv(b *testing.B) {
	tensor := newTensor()

	benchmarkOp(b, func() {
		tensor.Erfcinv()
	})
}

func BenchmarkErfinv(b *testing.B) {
	tensor := newTensor()

	benchmarkOp(b, func() {
		tensor.Erfinv()
	})
}

func BenchmarkExp(b *testing.B) {
	tensor := newTensor()

	benchmarkOp(b, func() {
		tensor.Exp()
	})
}

func BenchmarkExp2(b *testing.B) {
	tensor := newTensor()

	benchmarkOp(b, func() {
		tensor.Exp2()
	})
}

func BenchmarkExpm1(b *testing.B) {
	tensor := newTensor()

	benchmarkOp(b, func() {
		tensor.Expm1()
	})
}

func BenchmarkFMA(b *testing.B) {
	tensor := newTensor()

	benchmarkOp(b, func() {
		tensor.FMA(0, 1)
	})
}

func BenchmarkFloor(b *testing.B) {
	tensor := newTensor()

	benchmarkOp(b, func() {
		tensor.Floor()
	})
}

func BenchmarkGamma(b *testing.B) {
	tensor := newTensor()

	benchmarkOp(b, func() {
		tensor.Gamma()
	})
}

func BenchmarkIlogb(b *testing.B) {
	tensor := newTensor()

	benchmarkOp(b, func() {
		tensor.Ilogb()
	})
}

func BenchmarkInv(b *testing.B) {
	tensor := newTensor()

	benchmarkOp(b, func() {
		tensor.Inf()
	})
}

func BenchmarkJ0(b *testing.B) {
	tensor := newTensor()

	benchmarkOp(b, func() {
		tensor.J0()
	})
}

func BenchmarkJ1(b *testing.B) {
	tensor := newTensor()

	benchmarkOp(b, func() {
		tensor.J1()
	})
}

func BenchmarkJn(b *testing.B) {
	tensor := newTensor()

	benchmarkOp(b, func() {
		tensor.Jn(2)
	})
}

func BenchmarkLog(b *testing.B) {
	tensor := newTensor()

	benchmarkOp(b, func() {
		tensor.Log()
	})
}

func BenchmarkLog10(b *testing.B) {
	tensor := newTensor()

	benchmarkOp(b, func() {
		tensor.Log10()
	})
}

func BenchmarkLog1p(b *testing.B) {
	tensor := newTensor()

	benchmarkOp(b, func() {
		tensor.Log1p()
	})
}

func BenchmarkLog2(b *testing.B) {
	tensor := newTensor()

	benchmarkOp(b, func() {
		tensor.Log2()
	})
}

func BenchmarkLogb(b *testing.B) {
	tensor := newTensor()

	benchmarkOp(b, func() {
		tensor.Logb()
	})
}

func BenchmarkMod(b *testing.B) {
	tensor := newTensor()

	benchmarkOp(b, func() {
		tensor.Mod(2)
	})
}

func BenchmarkNaN(b *testing.B) {
	tensor := newTensor()

	benchmarkOp(b, func() {
		tensor.NaN()
	})
}

func BenchmarkNextafter(b *testing.B) {
	tensor := newTensor()

	benchmarkOp(b, func() {
		tensor.Nextafter(0)
	})
}

func BenchmarkNextafter32(b *testing.B) {
	tensor := newTensor()

	benchmarkOp(b, func() {
		tensor.Nextafter32(0)
	})
}

func BenchmarkPow(b *testing.B) {
	tensor := newTensor()

	benchmarkOp(b, func() {
		tensor.Pow(2)
	})
}

func BenchmarkPow10(b *testing.B) {
	tensor := newTensor()

	benchmarkOp(b, func() {
		tensor.Pow10(2)
	})
}

func BenchmarkRemainder(b *testing.B) {
	tensor := newTensor()

	benchmarkOp(b, func() {
		tensor.Remainder(2)
	})
}

func BenchmarkRound(b *testing.B) {
	tensor := newTensor()

	benchmarkOp(b, func() {
		tensor.Round()
	})
}

func BenchmarkRoundToEven(b *testing.B) {
	tensor := newTensor()

	benchmarkOp(b, func() {
		tensor.RoundToEven()
	})
}

func BenchmarkSin(b *testing.B) {
	tensor := newTensor()

	benchmarkOp(b, func() {
		tensor.Sin()
	})
}

func BenchmarkSinh(b *testing.B) {
	tensor := newTensor()

	benchmarkOp(b, func() {
		tensor.Sinh()
	})
}

func BenchmarkSqrt(b *testing.B) {
	tensor := newTensor()

	benchmarkOp(b, func() {
		tensor.Sqrt()
	})
}

func BenchmarkTan(b *testing.B) {
	tensor := newTensor()

	benchmarkOp(b, func() {
		tensor.Tan()
	})
}

func BenchmarkTanh(b *testing.B) {
	tensor := newTensor()

	benchmarkOp(b, func() {
		tensor.Tanh()
	})
}

func BenchmarkTrunc(b *testing.B) {
	tensor := newTensor()

	benchmarkOp(b, func() {
		tensor.Trunc()
	})
}

func BenchmarkY0(b *testing.B) {
	tensor := newTensor()

	benchmarkOp(b, func() {
		tensor.Y0()
	})
}

func BenchmarkY1(b *testing.B) {
	tensor := newTensor()

	benchmarkOp(b, func() {
		tensor.Y1()
	})
}

func BenchmarkYn(b *testing.B) {
	tensor := newTensor()

	benchmarkOp(b, func() {
		tensor.Yn(2)
	})
}