// Copyright © The Nune Author. All rights reserved.
// Use of this source logic, code, or documentation is governed by
// an MIT-style license that can be found in the LICENSE file.

package dispatch

import "github.com/vorduin/nune"

// CPD is the Central Processing Dispatch.
// It defines how the Tensor's data is stored, processed, and handled on CPU.
// This is Nune's default dispatch.
type CPD[T nune.Number] struct {
	data []T
}

// Dump sets the Tensor's data buffer on CPU to
// the given buffer.
func (c *CPD[T]) Dump(data []T) {
	c.data = data
}

// Load loads the Tensor's data buffer from RAM.
func (c *CPD[T]) Load() []T {
	return c.data
}

// Index returns the value at the given index from
// the Tensor's 1-dimensional data buffer on RAM.
func (c *CPD[T]) Index(idx int) T {
	return c.data[idx]
}

// SetIndex sets the index of the Tensor's 1-dimensional
// data buffer on RAM to the given value.
func (c *CPD[T]) SetIndex(idx int, value T) {
	c.data[idx] = value
}

// Slice returns a slice of the given bounds from
// the Tensor's 1-dimensional data buffer on RAM.
func (c *CPD[T]) Slice(start, end int) []T {
	return c.data[start:end]
}

// SetSlice sets the slice from the given bounds of
// the Tensor's 1-dimensional data buffer on RAM to the given slice.
func (c *CPD[T]) SetSlice(start, end int, s []T) {
	idx := 0
	for i := start; i < end; i++ {
		c.data[i] = s[idx]
		idx++
	}
}
