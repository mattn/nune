// Copyright © The Nune Author. All rights reserved.
// Use of this source logic, code, or documentation is governed by
// an MIT-style license that can be found in the LICENSE file.

package tensor

import (
	"github.com/vorduin/nune"
	"github.com/vorduin/slices"
)

// Cast casts a Tensor's underlying type to the given numeric type.
func Cast[T nune.Number, U nune.Number](t Tensor[U]) Tensor[T] {
	c := slices.WithLen[T](t.Numel())
	for i := 0; i < len(c); i++ {
		c[i] = T(t.dispatch.Index(i))
	}

	return Tensor[T]{
		dispatch: newDispatch(c),
		layout:   t.layout,
	}
}
