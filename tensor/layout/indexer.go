// Copyright © The Nune Author. All rights reserved.
// Use of this source logic, code, or documentation is governed by
// an MIT-style license that can be found in the LICENSE file.

package layout

// An Indexer is a tensor indexing scheme.
// All Indexer implementations must define their APIs to work
// accordingly along with the tensor's dispatch.
type Indexer interface {
	Shape() []int

	SetShape([]int)

	Strides() []int

	SetStrides() []int
}
