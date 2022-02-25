// Copyright © The Nune Author. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nune

import (
	"math"
	"reflect"

	"github.com/vorduin/slices"
)

// From returns a Tensor from the given backing - be it a numeric type,
// a sequence, or nested sequences - with the corresponding shape.
func From[T Number](b any) Tensor[T] {
	switch k := reflect.TypeOf(b).Kind(); k {
	case reflect.String:
		b = any([]byte(b.(string)))
		fallthrough
	case reflect.Array, reflect.Slice:
		v := reflect.ValueOf(b)

		c := make([]any, v.Len())
		for i := 0; i < v.Len(); i++ {
			c[i] = v.Index(i).Interface()
		}

		d, s, err := unwrapAny[T](c, []int{len(c)})
		if err != nil {
			if EnvConfig.Interactive {
				panic(err)
			} else {
				return Tensor[T]{
					Err: err,
				}
			}
		}

		return Tensor[T]{
			data:   d,
			shape:  s,
			stride: configStride(s),
		}
	default:
		if anyIsNumeric(b) {
			return Tensor[T]{
				data: anyToNumeric[T](b),
			}
		} else if c, ok := anyToTensor[T](b); ok {
			return c
		} else {
			if EnvConfig.Interactive {
				panic(ErrUnwrapBacking)
			} else {
				return Tensor[T]{
					Err: ErrUnwrapBacking,
				}
			}
		}
	}
}

// Full returns a Tensor full with the given value and
// satisfying the given shape.
func Full[T Number](x T, shape []int) Tensor[T] {
	err := verifyGoodShape(shape...)
	if err != nil {
		if EnvConfig.Interactive {
			panic(err)
		} else {
			return Tensor[T]{
				Err: err,
			}
		}
	}

	data := slices.WithLen[T](slices.Prod(shape))
	for i := 0; i < len(data); i++ {
		data[i] = T(x)
	}

	return Tensor[T]{
		data:   data,
		shape:  slices.Clone(shape),
		stride: configStride(shape),
	}
}

// FullLike returns a Tensor full with the given value and
// resembling the other Tensor's shape.
func FullLike[T Number, U Number](x T, other Tensor[U]) Tensor[T] {
	data := slices.WithLen[T](other.Numel())
	for i := 0; i < len(data); i++ {
		data[i] = T(x)
	}

	return Tensor[T]{
		data:   data,
		shape:  slices.Clone(other.shape),
		stride: configStride(other.shape),
	}
}

// Zeros returns a Tensor full with zeros and satisfying the given shape.
func Zeros[T Number](shape ...int) Tensor[T] {
	err := verifyGoodShape(shape...)
	if err != nil {
		if EnvConfig.Interactive {
			panic(err)
		} else {
			return Tensor[T]{
				Err: err,
			}
		}
	}

	return Tensor[T]{
		data:   slices.WithLen[T](int(slices.Prod(shape))),
		shape:  slices.Clone(shape),
		stride: configStride(shape),
	}
}

// ZerosLike returns a Tensor full with zeros and resembling the other
// Tensor's shape.
func ZerosLike[T Number, U Number](other Tensor[U]) Tensor[T] {
	return Tensor[T]{
		data:   slices.WithLen[T](other.Numel()),
		shape:  slices.Clone(other.shape),
		stride: configStride(other.shape),
	}
}

// Ones returns a Tensor full with ones and satisfying the given shape.
func Ones[T Number](shape ...int) Tensor[T] {
	return Full(T(1), shape)
}

// OnesLike returns a Tensor full with ones and resembling the other
// Tensor's shape.
func OnesLike[T Number, U Number](other Tensor[U]) Tensor[T] {
	return FullLike(T(1), other)
}

// Range returns a rank 1 Tensor on the interval [start, end),
// and with the given step-size.
func Range[T Number](start, end, step int) Tensor[T] {
	err := verifyGoodStep(step, start, end)
	if err != nil {
		if EnvConfig.Interactive {
			panic(err)
		} else {
			return Tensor[T]{
				Err: err,
			}
		}
	}

	d := math.Sqrt(math.Pow(float64(end-start), 2))   // distance
	l := int(math.Floor(d / math.Abs(float64(step)))) // length

	i := 0
	rng := slices.WithLen[T](l)
	for x := 0; x < l; x += 1 {
		rng[i] = T(start + x*step)
		i++
	}

	return Tensor[T]{
		data:   rng,
		shape:  []int{len(rng)},
		stride: configStride([]int{len(rng)}),
	}
}

// FromBuffer returns a Tensor with the given buffer set as its data buffer.
func FromBuffer[T Number](buf []T) Tensor[T] {
	err := verifyGoodShape(len(buf))
	if err != nil {
		if EnvConfig.Interactive {
			panic(err)
		} else {
			return Tensor[T]{
				Err: err,
			}
		}
	}

	return Tensor[T]{
		data:   buf,
		shape:  []int{len(buf)},
		stride: configStride([]int{len(buf)}),
	}
}
