// Copyright © The Nune Author. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tensor

import (
	"github.com/vorduin/nune"
	"github.com/vorduin/slices"
)

// Cast casts a Tensor's underlying type to the given numeric type.
func Cast[T nune.Number, V nune.Number](t Tensor[V]) Tensor[T] {
	if t.Err != nil {
		return Tensor[T]{
			Err: t.Err,
		}
	}

	dataBuf := t.Ravel()
	c := slices.WithLen[T](t.Numel())
	for i := 0; i < len(c); i++ {
		c[i] = T(dataBuf[i])
	}

	return Tensor[T]{
		data:  c,
		shape:   t.shape,
		stride: t.stride,
		offset: t.offset,
	}
}

// Clone clones the Tensor and its underlying view into its data buffer.
func (t Tensor[T]) Clone() Tensor[T] {
	if t.Err != nil {
		return t
	}

	return Tensor[T]{
		data: slices.Copy(t.Ravel()),
		shape: slices.Copy(t.shape),
		stride: slices.Copy(t.stride),
	}
}

// Reshape modifies the Tensor's indexing scheme.
// TODO: Check stride
func (t Tensor[T]) Reshape(shape ...int) Tensor[T] {
	if t.Err != nil {
		return t
	}

	if len(shape) == 0 && t.Numel() <= 1 {
		return Tensor[T]{
			data: t.data,
			offset: t.offset,
		}
	} else {
		err := verifyGoodShape(shape...)
		if err != nil {
			if nune.EnvConfig.Interactive {
				panic(err)
			} else {
				t.Err = err
				return t
			}
		}

		return Tensor[T]{
			data: t.data,
			shape: slices.Copy(shape),
			stride: configStride(shape),
			offset: t.offset,
		}
	}
}

// Index returns a view over an index of the Tensor.
// Multiple indices can be provided at the same time.
func (t Tensor[T]) Index(indices ...int) Tensor[T] {
	if t.Err != nil {
		return t
	}

	err := verifyArgsBounds(len(indices), t.Rank())
	if err != nil {
		if nune.EnvConfig.Interactive {
			panic(err)
		} else {
			t.Err = err
			return t
		}
	}

	for i, idx := range indices {
		err = verifyAxisBounds(idx, t.Size(i))
		if err != nil {
			if nune.EnvConfig.Interactive {
				panic(err)
			} else {
				t.Err = err
				return t
			}
		}
	}

	offset := t.offset

	for i, idx := range indices {
		offset += idx * t.stride[i]
	}

	return Tensor[T]{
		data:    t.data,
		shape:   slices.Copy(t.shape[len(indices):]),
		stride: slices.Copy(t.stride[len(indices):]),
		offset: offset,
	}
}

// Slice returns a view over a slice of the Tensor.
func (t Tensor[T]) Slice(start, end int) Tensor[T] {
	if t.Err != nil {
		return t
	}

	err := verifyGoodShape(t.shape...) // make sure Tensor rank is not 0
	if err != nil {
		if nune.EnvConfig.Interactive {
			panic(err)
		} else {
			t.Err = err
			return t
		}
	}

	err = verifyGoodInterval(start, end, [2]int{0, t.Size(0)})
	if err != nil {
		if nune.EnvConfig.Interactive {
			panic(err)
		} else {
			t.Err = err
			return t
		}
	}

	shape := slices.WithLen[int](len(t.shape))
	shape[0] = end - start
	copy(shape[1:], t.shape[1:])

	return Tensor[T]{
		data:    t.data,
		shape:   shape,
		stride: slices.Copy(t.stride),
		offset: t.offset+start*t.stride[0],
	}
}

// Broadcast broadcasts the Tensor to the given shape.
func (t Tensor[T]) Broadcast(shape ...int) Tensor[T] {
	if t.Err != nil {
		return t
	}

	if !t.Broadable(shape...) {
		if nune.EnvConfig.Interactive {
			panic(ErrNotBroadable)
		} else {
			t.Err = ErrNotBroadable
			return t
		}
	}

	var expandedShape []int

	if len(t.shape) < len(shape) {
		expandedShape = slices.WithLen[int](len(shape))
		for i := 0; i < len(shape)-len(t.shape); i++ {
			expandedShape[i] = 1
		}
		copy(expandedShape[len(shape)-len(t.shape):], t.shape)
	} else {
		expandedShape = t.shape
	}

	expandedStride := configStride(expandedShape)
	newStride := configStride(shape)

	data := slices.WithLen[T](int(slices.Prod(shape)))

	var expansion, stride int = 1, newStride[0]

	// This is around 20% slower on average the the shortened version
	// I also came up with, but this one generalizes correctly so...
	for axis := 0; axis < len(shape); axis++ {
		if expandedShape[axis] != shape[axis] {
			for i := 0; i < expansion; i++ {
				for j := 0; j < t.Numel()/expandedStride[axis]; j++ {
					for k := 0; k < shape[axis]; k++ {
						dstIdx := i * stride + j * shape[axis] + k * newStride[axis]
						srcIdx := j * expandedStride[axis]

						copy(data[dstIdx:dstIdx+newStride[axis]], t.Ravel()[srcIdx:srcIdx+expandedStride[axis]])
					}
				}
			}

			expansion *= shape[axis]
			stride = newStride[axis]
		}
	}

	return Tensor[T]{
		data: data,
		shape: slices.Copy(shape),
		stride: newStride,
	}
}

// Reverse reverses the order of the elements of the Tensor.
func (t Tensor[T]) Reverse() Tensor[T] {
	if t.Err != nil {
		return t
	}

	for i, j := 0, t.Numel()-1; i < j; i, j = i+1, j-1 {
		t.Ravel()[i], t.Ravel()[j] = t.Ravel()[j], t.Ravel()[i]
	}

	return t
}

// Permute permutes the Tensor's axes without changing the data. 
func (t Tensor[T]) Permute(axes ...int) Tensor[T] {
	if t.Err != nil {
		return t
	}

	err := verifyArgsBounds(len(axes), len(t.shape))
	if err != nil {
		if nune.EnvConfig.Interactive {
			panic(err)
		} else {
			t.Err = err
			return t
		}
	}

	shapeCopy := slices.Copy(t.shape)
	strideCopy := slices.Copy(t.stride)

	for i, axis := range axes {
		err := verifyAxisBounds(axis, len(t.shape))
		if err != nil {
			if nune.EnvConfig.Interactive {
				panic(err)
			} else {
				t.Err = err
				return t
			}
		}

		t.shape[i] = shapeCopy[axis]
		t.stride[i] = strideCopy[axis]
	}

	return t
}