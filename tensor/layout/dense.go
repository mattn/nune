// Copyright © The Nune Author. All rights reserved.
// Use of this source logic, code, or documentation is governed by
// an MIT-style license that can be found in the LICENSE file.

package layout

import "github.com/vorduin/slices"

// Dense implements a strided indexing scheme for the tensor package.
type Dense struct {
	shape   []int
	strides []int
}

// Configure takes a shape and configures the corresponding strides.
func (d Dense) Configure(shape []int) {
	d.shape = shape

	if len(shape) != 0 {
		strides := slices.WithLen[int](len(shape))
		strides[0] = slices.Prod(shape[1:])

		for i := 1; i < len(shape); i++ {
			strides[i] = strides[i-1] / shape[i]
		}

		d.strides = strides
	}
}

// Shape returns the layout's strided shape.
func (d Dense) Shape() []int {
	return d.shape
}

// SetShape modifies the layout's strided shape to the given shape.
func (d Dense) SetShape(shape []int) {
	d.shape = shape
}

// Strides returns the layout's strides.
func (d Dense) Strides() []int {
	return d.strides
}

// SetStrides modifies the layout's strides to the given strides.
func (d Dense) SetStrides(strides []int) {
	d.strides = strides
}

func (d Dense) Index(indices ...int) Indexer {
	den := new(Dense)
	shape := slices.Copy(d.shape[len(indices):])
	den.Configure(shape)

	return any(den).(Indexer)
}

func (d Dense) Slice(start, end int) Indexer {
	den := new(Dense)

	shape := slices.WithLen[int](len(d.shape))
	shape[0] = end - start
	copy(shape[1:], d.shape[1:])

	den.Configure(shape)

	return any(den).(Indexer)
}