// Copyright © The Nune Author. All rights reserved.
// Use of this source logic, code, or documentation is governed by
// an MIT-style license that can be found in the LICENSE file.

package nune

import (
	"runtime"
)

// EnvConfig holds Nune's environment configuration.
var EnvConfig = struct {
	Interactive bool
	NumCPU      int
}{
	Interactive: false,
	NumCPU:      runtime.NumCPU(),
}

// FmtConfig holds Nune's formatting configuration.
var FmtConfig = struct {
	Excerpt   int  // limit of the number of elements formatted
	Precision int  // limit of the number of decimals formatted
	Btoa      bool // convert bytes to ASCII
}{
	Excerpt:   6,
	Precision: 4,
	Btoa:      false,
}
