// Copyright © The Nune Author. All rights reserved.
// Use of this source logic, code, or documentation is governed by
// an MIT-style license that can be found in the LICENSE file.

package tensor

func Prod(s []int) int {
	p := 1
	for i := 0; i < len(s); i++ {
		p *= s[i]
	}
	return p
}