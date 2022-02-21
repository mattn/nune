// Copyright © The Nune Author. All rights reserved.
// Use of this source logic, code, or documentation is governed by
// an MIT-style license that can be found in the LICENSE file.

package dispatch

import "github.com/vorduin/nune"

// A Dispatcher takes care of storing and processing a Tensor's
// data on a specific type of hardware.
type Dispatcher[T nune.Number] interface {
	// Dump stores the Tensor's data on the hardware.
	Dump(data []T)

	// Load loads the Tensor's data from the hardware.
	Load() []T

	// Index returns the value at the given index from
	// the Tensor's 1-dimensional data buffer.
	Index(idx int) T

	// SetIndex sets the index of the Tensor's 1-dimensional
	// data buffer to the given value.
	SetIndex(idx int, value T)

	// Slice returns a slice of the given bounds from
	// the Tensor's 1-dimensional data buffer.
	Slice(start, end int) []T

	// SetSlice sets the slice from the given bounds of
	// the Tensor's 1-dimensional data buffer to the given slice.
	SetSlice(start, end int, s []T)
}
