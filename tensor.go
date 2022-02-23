// Copyright © The Nune Author. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nune

// A Tensor is a generic, n-dimensional numerical type.
type Tensor[T Number] struct {
	data          []T   // the tensor's data buffer
	shape, stride []int // the layout that holds the Tensor's indexing scheme
	offset        int   // the Tensor's view offset in the data buffer
	Err           error // holds the corresponding error when a Tensor operation fails
}
