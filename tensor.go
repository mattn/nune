// Copyright © The Nune Author. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nune

// Number is the set of all numeric types and their supersets.
type Number interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64 |
		~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 |
		~float32 | ~float64
}

// A Tensor is a generic, n-dimensional numerical type.
type Tensor[T Number] struct {
	data          []T   // the tensor's data buffer
	shape, stride []int // the layout that holds the Tensor's indexing scheme
	offset        int   // the Tensor's view offset in the data buffer
	Err           error // holds the corresponding error when a Tensor operation fails
}
