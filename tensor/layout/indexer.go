// Copyright © The Nune Author. All rights reserved.
// Use of this source logic, code, or documentation is governed by
// an MIT-style license that can be found in the LICENSE file.

package layout

// An Indexer is a tensor indexing scheme.
// All Indexer implementations must define their APIs to work
// accordingly along with the tensor's dispatch.
type Indexer interface {
	// Configure configures the indexing scheme of the Tensor.
	Configure(shape []int)

	// Shape returns the shape of the Tensor.
	Shape() []int

	// SetShape modifies the shape of the Tensor.
	SetShape(shape []int)

	// Strides returns the strides of the Tensor.
	Strides() []int

	// SetStrides modifies the strides of the Tensor.
	SetStrides(strides []int)
}
