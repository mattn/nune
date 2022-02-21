// Copyright © The Nune Author. All rights reserved.
// Use of this source logic, code, or documentation is governed by
// an MIT-style license that can be found in the LICENSE file.

package tensor

import (
	"github.com/vorduin/nune"
	"github.com/vorduin/slices"
)

// Cast casts a Tensor's underlying type to the given numeric type.
func Cast[T nune.Number, V nune.Number](t Tensor[V]) Tensor[T] {
	if t.Err != nil {
		return Tensor[T]{
			Err: t.Err,
		}
	}

	c := slices.WithLen[T](t.Numel())
	for i := 0; i < len(c); i++ {
		c[i] = T(t.data[i])
	}

	return Tensor[T]{
		data:    c,
		shape:   t.shape,
		strides: t.strides,
	}
}

// Index returns a view over an index of the Tensor.
// Multiple indices can be provided at the same time.
func (t Tensor[T]) Index(indices ...int) Tensor[T] {
	if t.Err != nil {
		return t
	}

	err := verifyArgsBounds(len(indices), t.Rank())
	if err != nil {
		if nune.EnvConfig.Interactive {
			panic(err)
		} else {
			return Tensor[T]{
				Err: err,
			}
		}
	}

	for i, idx := range indices {
		err = verifyAxisBounds(idx, t.Size(i))
		if err != nil {
			if nune.EnvConfig.Interactive {
				panic(err)
			} else {
				return Tensor[T]{
					Err: err,
				}
			}
		}
	}

	var offset int

	for i, idx := range indices {
		offset += idx * t.strides[i]
	}

	return Tensor[T]{
		data:    t.data[offset : offset+t.strides[len(indices)-1]],
		shape:   slices.Copy(t.shape[1:]),
		strides: slices.Copy(t.strides[1:]),
	}
}

// Slice returns a view over a slice of the Tensor.
func (t Tensor[T]) Slice(start, end int) Tensor[T] {
	if t.Err != nil {
		return t
	}

	err := verifyGoodShape(t.shape...) // make sure Tensor rank is not 0
	if err != nil {
		if nune.EnvConfig.Interactive {
			panic(err)
		} else {
			return Tensor[T]{
				Err: err,
			}
		}
	}

	err = verifyGoodInterval(start, end, [2]int{0, t.Size(0)})
	if err != nil {
		if nune.EnvConfig.Interactive {
			panic(err)
		} else {
			return Tensor[T]{
				Err: err,
			}
		}
	}

	shape := slices.WithLen[int](len(t.shape))
	shape[0] = end - start
	copy(shape[1:], t.shape[1:])

	return Tensor[T]{
		data:    t.data[start*t.strides[0] : end*t.strides[0]],
		shape:   shape,
		strides: slices.Copy(t.strides),
	}
}