// Copyright © The Nune Author. All rights reserved.
// Use of this source logic, code, or documentation is governed by
// an MIT-style license that can be found in the LICENSE file.

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
// Panic if axis is out of (0, rank) bounds.
func (t Tensor[T]) Size(axis int) int {
	err := verifyAxisBounds(axis, len(t.shape))
	if err != nil {
		panic(err)
	}

	return t.shape[axis]
}
