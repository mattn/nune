// Copyright © The Nune Author. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nune

import (
	"sync"
)

// HandlePwise processes a pointwise operation accordingly.
func handlePwise[T Number](in, out []T, f func(T) T, nCPU int) {
	var wg sync.WaitGroup

	for i := 0; i < nCPU; i++ {
		min := (i * len(in) / nCPU)
		max := ((i + 1) * len(in)) / nCPU

		wg.Add(1)
		go func(inBuf, outBuf []T) {
			for j := 0; j < len(inBuf); j++ {
				outBuf[j] = f(inBuf[j])
			}

			wg.Done()
		}(in[min:max], out[min:max])
	}

	wg.Wait()
}

// HandleEwise processes an elementwise operation accordingly.
func handleEwise[T Number](lhs, rhs, out []T, f func(T, T) T, nCPU int) {
	var wg sync.WaitGroup

	for i := 0; i < nCPU; i++ {
		min := (i * len(lhs) / nCPU)
		max := ((i + 1) * len(lhs)) / nCPU

		wg.Add(1)
		go func(lhsBuf, rhsBuf, outBuf []T) {
			for j := 0; j < len(lhsBuf); j++ {
				outBuf[j] = f(lhsBuf[j], rhsBuf[j])
			}

			wg.Done()
		}(lhs[min:max], rhs[min:max], out[min:max])
	}

	wg.Wait()
}

// HandleSwise processes a slice reduction operation accordingly.
func handleSwise[T Number](in []T, out *T, f func([]T) T, nCPU int) {
	outBuf := make([]T, 0, nCPU)
	ch := make(chan T, nCPU)

	for i := 0; i < nCPU; i++ {
		min := (i * len(in) / nCPU)
		max := ((i + 1) * len(in)) / nCPU

		go func(inBuf []T, c chan<- T) {
			c <- f(inBuf)
		}(in[min:max], ch)
	}

	for i := 0; i < nCPU; i++ {
		outBuf = append(outBuf, <-ch)
	}

	*out = f(outBuf)
}