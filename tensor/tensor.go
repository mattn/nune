// Copyright © The Nune Author. All rights reserved.
// Use of this source logic, code, or documentation is governed by
// an MIT-style license that can be found in the LICENSE file.

package tensor

import (
	"github.com/vorduin/nune"
	"github.com/vorduin/nune/tensor/dispatch"
	"github.com/vorduin/nune/tensor/layout"
)

// A Tensor is a generic, n-dimensional numerical type.
type Tensor[T nune.Number] struct {
	dispatch *dispatch.Dispatcher[T] // the dispatch that processes the Tensor's data
	layout   *layout.Indexer         // the layout that holds the Tensor's indexing scheme
	Err      error                   // holds the corresponding error when a Tensor operation fails
}
