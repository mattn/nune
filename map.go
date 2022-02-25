// Copyright © The Nune Author. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nune

import (
	"sync"
	"math"
)

// handleMap processes a pointwise operation accordingly.
func handleMap[T Number](in, out []T, f func(T) T, nCPU int) {
	var wg sync.WaitGroup

	for i := 0; i < nCPU; i++ {
		min := (i * len(in) / nCPU)
		max := ((i + 1) * len(in)) / nCPU

		wg.Add(1)
		go func(inBuf, outBuf []T) {
			for j := 0; j < len(inBuf); j++ {
				outBuf[j] = f(inBuf[j])
			}

			wg.Done()
		}(in[min:max], out[min:max])
	}

	wg.Wait()
}

// Map performs a pointwise operation over the elements of this Tensor.
func (t Tensor[T]) Map(f func(T) T) Tensor[T] {
	if t.Err != nil {
		if EnvConfig.Interactive {
			panic(t.Err)
		} else {
			return t
		}
	}

	handleMap(t.Ravel(), t.Ravel(), f, configCPU(t.Numel()))

	return t
}

// Abs computes the absolute value of each element in the Tensor.
func (t Tensor[T]) Abs() Tensor[T] {
	return t.Map(func(x T) T {
		return T(math.Abs(float64(x)))
	})
}

// Acos computes the arccosine, in radians, of each element in the Tensor.
func (t Tensor[T]) Acos() Tensor[T] {
	return t.Map(func(x T) T {
		return T(math.Acos(float64(x)))
	})
}

// Acosh computes the inverse hyperbolic cosine of each element in the Tensor.
func (t Tensor[T]) Acosh() Tensor[T] {
	return t.Map(func(x T) T {
		return T(math.Acosh(float64(x)))
	})
}

// Asin computes the arcsine, in radians, of each element in the Tensor.
func (t Tensor[T]) Asin() Tensor[T] {
	return t.Map(func(x T) T {
		return T(math.Asin(float64(x)))
	})
}

// Asinh computes the inverse hyperbolic sine of each element in the Tensor.
func (t Tensor[T]) Asinh() Tensor[T] {
	return t.Map(func(x T) T {
		return T(math.Asinh(float64(x)))
	})
}

// Atan computes the arctangent, in radians, of each element in the Tensor.
func (t Tensor[T]) Atan() Tensor[T] {
	return t.Map(func(x T) T {
		return T(math.Atan(float64(x)))
	})
}

// Atan2 computes the arc tangent of y/x, where x is each element in the Tensor,
// using the signs of the two to determine the quadrant of the resulting value.
func (t Tensor[T]) Atan2(y float64) Tensor[T] {
	return t.Map(func(x T) T {
		return T(math.Atan2(y, float64(x)))
	})
}

// Atanh computes the inverse hyperbolic tangent of each element in the Tensor.
func (t Tensor[T]) Atanh() Tensor[T] {
	return t.Map(func(x T) T {
		return T(math.Atanh(float64(x)))
	})
}

// Cbrt computes the cubic root of each element in the Tensor.
func (t Tensor[T]) Cbrt() Tensor[T] {
	return t.Map(func(x T) T {
		return T(math.Cbrt(float64(x)))
	})
}

// Ceil computes the least integer value great than or equal to x,
// where x is each element in the Tensor.
func (t Tensor[T]) Ceil() Tensor[T] {
	return t.Map(func(x T) T {
		return T(math.Ceil(float64(x)))
	})
}

// Copysign computes a value with the magnitude of x and
// the sign of y, where x is each element in the Tensor.
func (t Tensor[T]) Copysign(y float64) Tensor[T] {
	return t.Map(func(x T) T {
		return T(math.Copysign(float64(x), y))
	})
}

// Cos computes the cosine of each radian element of the Tensor.
func (t Tensor[T]) Cos() Tensor[T] {
	return t.Map(func(x T) T {
		return T(math.Cos(float64(x)))
	})
}

// Cosh computes the hyperbolic cosine of each element in the Tensor.
func (t Tensor[T]) Cosh() Tensor[T] {
	return t.Map(func(x T) T {
		return T(math.Cosh(float64(x)))
	})
}

// Dim computes the maximum of x-y or 0, where x is each
// element in the Tensor.
func (t Tensor[T]) Dim(y float64) Tensor[T] {
	return t.Map(func(x T) T {
		return T(math.Dim(float64(x), y))
	})
}

// Erf computes the error function of each element in the Tensor.
func (t Tensor[T]) Erf() Tensor[T] {
	return t.Map(func(x T) T {
		return T(math.Erf(float64(x)))
	})
}

// Erfc computes the complementary error function of each
// element in the Tensor.
func (t Tensor[T]) Erfc() Tensor[T] {
	return t.Map(func(x T) T {
		return T(math.Erfc(float64(x)))
	})
}

// Erfcinv computes the inverse complementary error function
// for each element in the Tensor.
func (t Tensor[T]) Erfcinv() Tensor[T] {
	return t.Map(func(x T) T {
		return T(math.Erfcinv(float64(x)))
	})
}

// Erfinv computes the inverse error function for each
// element in the Tensor.
func (t Tensor[T]) Erfinv() Tensor[T] {
	return t.Map(func(x T) T {
		return T(math.Erfinv(float64(x)))
	})
}

// Exp computes the base-e exponential of each element in the Tensor.
func (t Tensor[T]) Exp() Tensor[T] {
	return t.Map(func(x T) T {
		return T(math.Exp(float64(x)))
	})
}

// Exp2 computes the base-2 exponential of each element in the Tensor.
func (t Tensor[T]) Exp2() Tensor[T] {
	return t.Map(func(x T) T {
		return T(math.Exp2(float64(x)))
	})
}

// Expm1 computes the base-e exponential of each element in the Tensor minus 1.
// It is more accurate than exp(x) - 1 when the elements are near zero.
func (t Tensor[T]) Expm1() Tensor[T] {
	return t.Map(func(x T) T {
		return T(math.Expm1(float64(x)))
	})
}

// FMA computes x * y + z, where x is each element in the Tensor,
// with only one rounding.
// (That is, FMA returns the fused multiply-add of x, y, and z.)
func (t Tensor[T]) FMA(y, z float64) Tensor[T] {
	return t.Map(func(x T) T {
		return T(math.FMA(float64(x), y, z))
	})
}

// Floor computes the greatest integer value less than or equal to
// each element in the Tensor.
func (t Tensor[T]) Floor() Tensor[T] {
	return t.Map(func(x T) T {
		return T(math.Floor(float64(x)))
	})
}

// Gamma computes the Gamma function of each element in the Tensor.
func (t Tensor[T]) Gamma() Tensor[T] {
	return t.Map(func(x T) T {
		return T(math.Gamma(float64(x)))
	})
}

// Ilogb computes the binary exponent of each element in the Tensor
// as an integer.
func (t Tensor[T]) Ilogb() Tensor[T] {
	return t.Map(func(x T) T {
		return T(math.Ilogb(float64(x)))
	})
}

// Inf computes positive infinity if x >= 0, negative infinity if x < 0,
// where x is each element in the Tensor.
func (t Tensor[T]) Inf() Tensor[T] {
	return t.Map(func(x T) T {
		return T(math.Inf(int(x)))
	})
}

// J0 computes the order-zero Bessel function of the first kind
// for each element in the Tensor.
func (t Tensor[T]) J0() Tensor[T] {
	return t.Map(func(x T) T {
		return T(math.J0(float64(x)))
	})
}

// J1 computes the order-one Bessel function of the first kind
// for each element in the Tensor.
func (t Tensor[T]) J1() Tensor[T] {
	return t.Map(func(x T) T {
		return T(math.J1(float64(x)))
	})
}

// Jn computes the order-n Bessel function of the first kind
// for each element in the Tensor.
func (t Tensor[T]) Jn(n int) Tensor[T] {
	return t.Map(func(x T) T {
		return T(math.Jn(n, float64(x)))
	})
}

// Log computes the natural logarithm for each element in the Tensor.
func (t Tensor[T]) Log() Tensor[T] {
	return t.Map(func(x T) T {
		return T(math.Log(float64(x)))
	})
}

// Log10 computes the decimal logarithm for each element in the Tensor.
func (t Tensor[T]) Log10() Tensor[T] {
	return t.Map(func(x T) T {
		return T(math.Log10(float64(x)))
	})
}

// Log1p computes the natural logarithm of 1 plus its argument x,
// where x is each element in the Tensor. It is more accurate
// than log(1 + x) when x is near zero.
func (t Tensor[T]) Log1p() Tensor[T] {
	return t.Map(func(x T) T {
		return T(math.Log1p(float64(x)))
	})
}

// Log2 computes the binary logarithm of each element in the Tensor.
func (t Tensor[T]) Log2() Tensor[T] {
	return t.Map(func(x T) T {
		return T(math.Log2(float64(x)))
	})
}

// Logb computes the binary exponent of each element in the Tensor.
func (t Tensor[T]) Logb() Tensor[T] {
	return t.Map(func(x T) T {
		return T(math.Logb(float64(x)))
	})
}

// Mod computes the floating-point remainder of x/y, where x is each element
// in the Tensor. The magnitude of the result is less than y and
// its sign agrees with that of x. 
func (t Tensor[T]) Mod(y float64) Tensor[T] {
	return t.Map(func(x T) T {
		return T(math.Mod(float64(x), y))
	})
}

// NaN computes an IEEE 754 “not-a-number” value for each element
// in the Tensor. Using this function is discouraged.
func (t Tensor[T]) NaN() Tensor[T] {
	return t.Map(func(x T) T {
		return T(math.NaN())
	})
}

// Nextafter computes the next representable float64 value after x towards y,
// where x is each element in the Tensor.
func (t Tensor[T]) Nextafter(y float64) Tensor[T] {
	return t.Map(func(x T) T {
		return T(math.Nextafter(float64(x), y))
	})
}

// Nextafter32 computes the next representable float32 value after x towards y,
// where x is each element in the Tensor.
func (t Tensor[T]) Nextafter32(y float32) Tensor[T] {
	return t.Map(func(x T) T {
		return T(math.Nextafter32(float32(x), y))
	})
}

// Pow computes the base-x exponential of y, where x is each
// element in the Tensor.
func (t Tensor[T]) Pow(y float64) Tensor[T] {
	return t.Map(func(x T) T {
		return T(math.Pow(float64(x), y))
	})
}

// Pow10 computes the base-10 exponential of n, for each element
// in the Tensor.
func (t Tensor[T]) Pow10(n int) Tensor[T] {
	return t.Map(func(x T) T {
		return T(math.Pow10(n))
	})
}

// Remainder computes the IEEE 754 floating-point remainder of x/y,
// where x is each element in the Tensor.
func (t Tensor[T]) Remainder(y float64) Tensor[T] {
	return t.Map(func(x T) T {
		return T(math.Remainder(float64(x), y))
	})
}

// Round computes the nearest integer, rounding half away from zero,
// for each element in the Tensor.
func (t Tensor[T]) Round() Tensor[T] {
	return t.Map(func(x T) T {
		return T(math.Round(float64(x)))
	})
}

// Round to even computes the nearest integer, rounding ties to even,
// for each element in the Tensor.
func (t Tensor[T]) RoundToEven() Tensor[T] {
	return t.Map(func(x T) T {
		return T(math.RoundToEven(float64(x)))
	})
}

// Sin computes the sine of each radian element of the Tensor.
func (t Tensor[T]) Sin() Tensor[T] {
	return t.Map(func(x T) T {
		return T(math.Sin(float64(x)))
	})
}

// Sinh computes the hyperbolic sine of each element in the Tensor.
func (t Tensor[T]) Sinh() Tensor[T] {
	return t.Map(func(x T) T {
		return T(math.Sinh(float64(x)))
	})
}

// Sqrt computes the square root of each element in the Tensor.
func (t Tensor[T]) Sqrt() Tensor[T] {
	return t.Map(func(x T) T {
		return T(math.Sqrt(float64(x)))
	})
}

// Tan computes the tangent of each radian element of the Tensor.
func (t Tensor[T]) Tan() Tensor[T] {
	return t.Map(func(x T) T {
		return T(math.Tan(float64(x)))
	})
}

// Tanh computes the hyperbolic tangent of each radian element of the Tensor.
func (t Tensor[T]) Tanh() Tensor[T] {
	return t.Map(func(x T) T {
		return T(math.Tanh(float64(x)))
	})
}

// Trunc computes the integer value of each element in the Tensor.
func (t Tensor[T]) Trunc() Tensor[T] {
	return t.Map(func(x T) T {
		return T(math.Trunc(float64(x)))
	})
}

// Y0 computes the order-zero Bessel function of the second kind
// of each element of the Tensor.
func (t Tensor[T]) Y0() Tensor[T] {
	return t.Map(func(x T) T {
		return T(math.Y0(float64(x)))
	})
}

// Y1 computes the order-one Bessel function of the second kind
// of each element in the Tensor.
func (t Tensor[T]) Y1() Tensor[T] {
	return t.Map(func(x T) T {
		return T(math.Y1(float64(x)))
	})
}

// Yn computes the order-n Bessel function of the second kind
// of each element in the Tensor.
func (t Tensor[T]) Yn(n int) Tensor[T] {
	return t.Map(func(x T) T {
		return T(math.Yn(n, float64(x)))
	})
}