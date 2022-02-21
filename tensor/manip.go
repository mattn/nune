// Copyright © The Nune Author. All rights reserved.
// Use of this source logic, code, or documentation is governed by
// an MIT-style license that can be found in the LICENSE file.

package tensor

import (
	"github.com/vorduin/nune"
	"github.com/vorduin/nune/tensor/dispatch"
)

// Cast casts a Tensor's underlying type to the given numeric type.
func Cast[T nune.Number, V nune.Number](t Tensor[V]) Tensor[T] {
	return Tensor[T]{
		dispatch: t.dispatch.Cast(T(0)).(dispatch.Dispatcher[T]),
		layout: t.layout,
	}
}

// Index returns a view over an index of the Tensor.
// Multiple indices can be provided at the same time.
func (t Tensor[T]) Index(indices ...int) Tensor[T] {
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
		offset += idx * t.layout.Strides()[i]
	}

	return Tensor[T]{
		dispatch: t.dispatch.Slice(offset, offset+t.layout.Strides()[len(indices)-1]),
		layout: t.layout.Index(indices...),
	}
}

// Slice returns a view over a slice of the Tensor.
func (t Tensor[T]) Slice(start, end int) Tensor[T] {
	err := verifyGoodShape(t.layout.Shape()...) // make sure Tensor rank is not 0
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

	return Tensor[T]{
		dispatch: t.dispatch.Slice(start*t.layout.Strides()[0], end*t.layout.Strides()[0]),
		layout: t.layout.Slice(start, end),
	}
}