// Copyright © The Nune Author. All rights reserved.
// Use of this source logic, code, or documentation is governed by
// an MIT-style license that can be found in the LICENSE file.

package tensor

import (
	"math"
	"reflect"

	"github.com/vorduin/nune"
	"github.com/vorduin/slices"
)

// From returns a Tensor from the given backing - be it a numeric type,
// a sequence, or nested sequences - with the corresponding shape.
func From[T nune.Number](b any) Tensor[T] {
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
			if nune.EnvConfig.Interactive {
				panic(err)
			} else {
				return Tensor[T]{
					Err: err,
				}
			}
		}

		return Tensor[T]{
			data:    d,
			shape:   s,
			strides: configStrides(s),
		}
	default:
		if anyIsNumeric(b) {
			return Tensor[T]{
				data: anyToNumeric[T](b),
			}
		} else if c, ok := anyToTensor[T](b); ok {
			return c
		} else {
			if nune.EnvConfig.Interactive {
				panic(ErrUnwrapBacking)
			} else {
				return Tensor[T]{
					Err: ErrUnwrapBacking,
				}
			}
		}
	}
}

// Full returns a Tensor filled with the given value and
// satisfying the given shape.
func Full[T nune.Number](x T, shape []int) Tensor[T] {
	err := verifyGoodShape(shape...)
	if err != nil {
		if nune.EnvConfig.Interactive {
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
		data:    data,
		shape:   slices.Copy(shape),
		strides: configStrides(shape),
	}
}

// Zeros returns a Tensor filled with zeros and satisfying the given shape.
func Zeros[T nune.Number](shape ...int) Tensor[T] {
	return Full(T(0), shape)
}

// Ones returns a Tensor filled with ones and satisfying the given shape.
func Ones[T nune.Number](shape ...int) Tensor[T] {
	return Full(T(1), shape)
}

// Range returns a rank 1 Tensor on the interval [start, end),
// and with the given step-size.
func Range[T nune.Number](start, end, step int) Tensor[T] {
	err := verifyGoodStep(step, start, end)
	if err != nil {
		if nune.EnvConfig.Interactive {
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
		data:    rng,
		shape:   []int{len(rng)},
		strides: configStrides([]int{len(rng)}),
	}
}
