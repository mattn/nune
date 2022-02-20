// Copyright © The Nune Author. All rights reserved.
// Use of this source logic, code, or documentation is governed by
// an MIT-style license that can be found in the LICENSE file.

package dispatch

import "github.com/vorduin/nune"

type Dispatcher[T nune.Number] interface {
	Dump(data []T)

	Load() []T

	Index(idx int) T

	SetIndex(idx int, value T)

	Slice(start, end int)

	SetSlice(start, end int, s []T)
}
