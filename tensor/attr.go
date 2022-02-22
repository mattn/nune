// Copyright © The Nune Author. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tensor

import (
	"github.com/vorduin/slices"
)

// Ravel returns a copy of the Tensor's data buffer.
func (t Tensor[T]) Ravel() []T {
	return slices.Copy(t.data)
}

// Numel returns the number of elements in the Tensor's data buffer.
func (t Tensor[T]) Numel() int {
	return len(t.data)
}

// Rank returns the Tensor's rank
// (the number of axes in the Tensor's shape).
func (t Tensor[T]) Rank() int {
	return len(t.shape)
}

// Shape returns a copy of the Tensor's shape.
func (t Tensor[T]) Shape() []int {
	return slices.Copy(t.shape)
}

// Strides returns a copy of the Tensor's strides.
func (t Tensor[T]) Strides() []int {
	return slices.Copy(t.strides)
}

// Size returns the Tensor's number of dimensions at
// the given axis.
// Panics if axis is out of (0, rank) bounds.
func (t Tensor[T]) Size(axis int) int {
	err := verifyAxisBounds(axis, len(t.shape))
	if err != nil {
		panic(err)
	}

	return t.shape[axis]
}

// Broadable returns whether or not the Tensor can be
// broadcast to the given shape.
func (t *Tensor[T]) Broadable(shape ...int) bool {
	err := verifyGoodShape(shape...)
	if err != nil {
		return false
	}

	err = verifyArgsBounds(len(t.shape), len(shape))
	if err != nil {
		return false
	}

	var s []int

	if len(t.shape) < len(shape) {
		s = slices.WithLen[int](len(shape))
		for i := 0; i < len(shape)-len(t.shape); i++ {
			s[i] = 1
		}
		copy(s[len(shape)-len(t.shape):], t.shape)
	} else {
		s = t.shape
	}

	for i := 0; i < len(shape); i++ {
		if s[i] != shape[i] && s[i] != 1 {
			return false
		}
	}

	return true
}