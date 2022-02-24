// Copyright © The Nune Author. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nune

import (
	"math"
)

// Pointwise performs a pointwise operation
// over the elements of this Tensor.
func (t Tensor[T]) Pointwise(f func(T) T) Tensor[T] {
	if t.Err != nil {
		if EnvConfig.Interactive {
			panic(t.Err)
		} else {
			return t
		}
	}

	handlePointwise(t.Ravel(), t.Ravel(), f, EnvConfig.NumCPU)

	return t
}

func (t Tensor[T]) Abs() Tensor[T] {
	return t.Pointwise(func(x T) T {
		return T(math.Abs(float64(x)))
	})
}

func (t Tensor[T]) Acos() Tensor[T] {
	return t.Pointwise(func(x T) T {
		return T(math.Acos(float64(x)))
	})
}

func (t Tensor[T]) Acosh() Tensor[T] {
	return t.Pointwise(func(x T) T {
		return T(math.Acosh(float64(x)))
	})
}

func (t Tensor[T]) Asin() Tensor[T] {
	return t.Pointwise(func(x T) T {
		return T(math.Asin(float64(x)))
	})
}

func (t Tensor[T]) Asinh() Tensor[T] {
	return t.Pointwise(func(x T) T {
		return T(math.Asinh(float64(x)))
	})
}

func (t Tensor[T]) Atan() Tensor[T] {
	return t.Pointwise(func(x T) T {
		return T(math.Atan(float64(x)))
	})
}

func (t Tensor[T]) Atan2(y float64) Tensor[T] {
	return t.Pointwise(func(x T) T {
		return T(math.Atan2(y, float64(x)))
	})
}

func (t Tensor[T]) Atanh() Tensor[T] {
	return t.Pointwise(func(x T) T {
		return T(math.Atanh(float64(x)))
	})
}

func (t Tensor[T]) Cbrt() Tensor[T] {
	return t.Pointwise(func(x T) T {
		return T(math.Cbrt(float64(x)))
	})
}

func (t Tensor[T]) Ceil() Tensor[T] {
	return t.Pointwise(func(x T) T {
		return T(math.Ceil(float64(x)))
	})
}

func (t Tensor[T]) Copysign(y float64) Tensor[T] {
	return t.Pointwise(func(x T) T {
		return T(math.Copysign(float64(x), y))
	})
}

func (t Tensor[T]) Cos() Tensor[T] {
	return t.Pointwise(func(x T) T {
		return T(math.Cos(float64(x)))
	})
}

func (t Tensor[T]) Cosh() Tensor[T] {
	return t.Pointwise(func(x T) T {
		return T(math.Cosh(float64(x)))
	})
}

func (t Tensor[T]) Dim(y float64) Tensor[T] {
	return t.Pointwise(func(x T) T {
		return T(math.Dim(float64(x), y))
	})
}

func (t Tensor[T]) Erf() Tensor[T] {
	return t.Pointwise(func(x T) T {
		return T(math.Erf(float64(x)))
	})
}

func (t Tensor[T]) Erfc() Tensor[T] {
	return t.Pointwise(func(x T) T {
		return T(math.Erfc(float64(x)))
	})
}

func (t Tensor[T]) Erfcinv() Tensor[T] {
	return t.Pointwise(func(x T) T {
		return T(math.Erfcinv(float64(x)))
	})
}

func (t Tensor[T]) Erfinv() Tensor[T] {
	return t.Pointwise(func(x T) T {
		return T(math.Erfinv(float64(x)))
	})
}

func (t Tensor[T]) Exp() Tensor[T] {
	return t.Pointwise(func(x T) T {
		return T(math.Exp(float64(x)))
	})
}

func (t Tensor[T]) Exp2() Tensor[T] {
	return t.Pointwise(func(x T) T {
		return T(math.Exp2(float64(x)))
	})
}

func (t Tensor[T]) Expm1() Tensor[T] {
	return t.Pointwise(func(x T) T {
		return T(math.Expm1(float64(x)))
	})
}

func (t Tensor[T]) FMA(y, z float64) Tensor[T] {
	return t.Pointwise(func(x T) T {
		return T(math.FMA(float64(x), y, z))
	})
}

func (t Tensor[T]) Floor() Tensor[T] {
	return t.Pointwise(func(x T) T {
		return T(math.Floor(float64(x)))
	})
}

func (t Tensor[T]) Gamma() Tensor[T] {
	return t.Pointwise(func(x T) T {
		return T(math.Gamma(float64(x)))
	})
}

func (t Tensor[T]) Ilogb() Tensor[T] {
	return t.Pointwise(func(x T) T {
		return T(math.Ilogb(float64(x)))
	})
}

func (t Tensor[T]) Inf() Tensor[T] {
	return t.Pointwise(func(x T) T {
		return T(math.Inf(int(x)))
	})
}

func (t Tensor[T]) J0() Tensor[T] {
	return t.Pointwise(func(x T) T {
		return T(math.J0(float64(x)))
	})
}

func (t Tensor[T]) J1() Tensor[T] {
	return t.Pointwise(func(x T) T {
		return T(math.J1(float64(x)))
	})
}

func (t Tensor[T]) Jn(n int) Tensor[T] {
	return t.Pointwise(func(x T) T {
		return T(math.Jn(n, float64(x)))
	})
}

func (t Tensor[T]) Ldexp(exp int) Tensor[T] {
	return t.Pointwise(func(x T) T {
		return T(math.Ldexp(float64(x), exp))
	})
}

func (t Tensor[T]) Log() Tensor[T] {
	return t.Pointwise(func(x T) T {
		return T(math.Log(float64(x)))
	})
}

func (t Tensor[T]) Log10() Tensor[T] {
	return t.Pointwise(func(x T) T {
		return T(math.Log10(float64(x)))
	})
}

func (t Tensor[T]) Log1p() Tensor[T] {
	return t.Pointwise(func(x T) T {
		return T(math.Log1p(float64(x)))
	})
}

func (t Tensor[T]) Log2() Tensor[T] {
	return t.Pointwise(func(x T) T {
		return T(math.Log2(float64(x)))
	})
}

func (t Tensor[T]) Logb() Tensor[T] {
	return t.Pointwise(func(x T) T {
		return T(math.Logb(float64(x)))
	})
}

func (t Tensor[T]) Mod(y float64) Tensor[T] {
	return t.Pointwise(func(x T) T {
		return T(math.Mod(float64(x), y))
	})
}

func (t Tensor[T]) NaN() Tensor[T] {
	return t.Pointwise(func(x T) T {
		return T(math.NaN())
	})
}

func (t Tensor[T]) Pow(y float64) Tensor[T] {
	return t.Pointwise(func(x T) T {
		return T(math.Pow(float64(x), y))
	})
}

func (t Tensor[T]) Pow10(n int) Tensor[T] {
	return t.Pointwise(func(x T) T {
		return T(math.Pow10(n))
	})
}

func (t Tensor[T]) Remainder(y float64) Tensor[T] {
	return t.Pointwise(func(x T) T {
		return T(math.Remainder(float64(x), y))
	})
}

func (t Tensor[T]) Round() Tensor[T] {
	return t.Pointwise(func(x T) T {
		return T(math.Round(float64(x)))
	})
}

func (t Tensor[T]) RoundToEven() Tensor[T] {
	return t.Pointwise(func(x T) T {
		return T(math.RoundToEven(float64(x)))
	})
}

func (t Tensor[T]) Sin() Tensor[T] {
	return t.Pointwise(func(x T) T {
		return T(math.Sin(float64(x)))
	})
}

func (t Tensor[T]) Sinh() Tensor[T] {
	return t.Pointwise(func(x T) T {
		return T(math.Sinh(float64(x)))
	})
}

func (t Tensor[T]) Sqrt() Tensor[T] {
	return t.Pointwise(func(x T) T {
		return T(math.Sqrt(float64(x)))
	})
}

func (t Tensor[T]) Tan() Tensor[T] {
	return t.Pointwise(func(x T) T {
		return T(math.Tan(float64(x)))
	})
}

func (t Tensor[T]) Tanh() Tensor[T] {
	return t.Pointwise(func(x T) T {
		return T(math.Tanh(float64(x)))
	})
}

func (t Tensor[T]) Trunc() Tensor[T] {
	return t.Pointwise(func(x T) T {
		return T(math.Trunc(float64(x)))
	})
}

func (t Tensor[T]) Y0() Tensor[T] {
	return t.Pointwise(func(x T) T {
		return T(math.Y0(float64(x)))
	})
}

func (t Tensor[T]) Y1() Tensor[T] {
	return t.Pointwise(func(x T) T {
		return T(math.Y1(float64(x)))
	})
}

func (t Tensor[T]) Yn(n int) Tensor[T] {
	return t.Pointwise(func(x T) T {
		return T(math.Yn(n, float64(x)))
	})
}