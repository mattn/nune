// Copyright © The Nune Author. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nune

import (
	"errors"

	"github.com/vorduin/slices"
)

// Cast casts a Tensor's underlying type to the given numeric type.
func Cast[T Number, V Number](t Tensor[V]) Tensor[T] {
	if t.Err != nil {
		if EnvConfig.Interactive {
			panic(t.Err)
		} else {
			return Tensor[T]{
				Err: t.Err,
			}
		}
	}

	dataBuf := t.Ravel()
	c := slices.WithLen[T](t.Numel())
	for i := 0; i < len(c); i++ {
		c[i] = T(dataBuf[i])
	}

	return Tensor[T]{
		data:   c,
		shape:  t.shape,
		stride: t.stride,
		offset: t.offset,
	}
}

// Clone clones the Tensor and its underlying view into its data buffer.
func (t Tensor[T]) Clone() Tensor[T] {
	if t.Err != nil {
		if EnvConfig.Interactive {
			panic(t.Err)
		} else {
			return t
		}
	}

	return Tensor[T]{
		data:   slices.Clone(t.Ravel()),
		shape:  slices.Clone(t.shape),
		stride: slices.Clone(t.stride),
	}
}

// Reshape modifies the Tensor's indexing scheme.
func (t Tensor[T]) Reshape(shape ...int) Tensor[T] {
	if t.Err != nil {
		if EnvConfig.Interactive {
			panic(t.Err)
		} else {
			return t
		}
	}

	if len(shape) == 0 && t.Numel() <= 1 {
		return Tensor[T]{
			data:   t.data,
			offset: t.offset,
		}
	} else {
		err := verifyGoodShape(shape...)
		if err != nil {
			if EnvConfig.Interactive {
				panic(err)
			} else {
				t.Err = err
				return t
			}
		}
		
		newstride := slices.WithLen[int](len(shape))
		if len(shape) <= len(t.shape) {
			copy(newstride, t.stride[len(t.stride)-len(shape):])
		} else {
			copy(newstride[len(shape)-len(t.stride):], t.stride)
			for i := len(shape)-len(t.stride)-1; i >= 0; i-- {
				newstride[i] = shape[i+1] * newstride[i+1]
			}
		}

		return Tensor[T]{
			data:   t.data,
			shape:  slices.Clone(shape),
			stride: newstride,
			offset: t.offset,
		}
	}
}

// Index returns a view over an index of the Tensor.
// Multiple indices can be provided at the same time.
func (t Tensor[T]) Index(indices ...int) Tensor[T] {
	if t.Err != nil {
		if EnvConfig.Interactive {
			panic(t.Err)
		} else {
			return t
		}
	}

	err := verifyArgsBounds(len(indices), t.Rank())
	if err != nil {
		if EnvConfig.Interactive {
			panic(err)
		} else {
			t.Err = err
			return t
		}
	}

	for i, idx := range indices {
		err = verifyAxisBounds(idx, t.Size(i))
		if err != nil {
			if EnvConfig.Interactive {
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
		data:   t.data,
		shape:  slices.Clone(t.shape[len(indices):]),
		stride: slices.Clone(t.stride[len(indices):]),
		offset: offset,
	}
}

// Slice returns a view over a slice of the Tensor.
func (t Tensor[T]) Slice(start, end int) Tensor[T] {
	if t.Err != nil {
		if EnvConfig.Interactive {
			panic(t.Err)
		} else {
			return t
		}
	}

	err := verifyGoodShape(t.shape...) // make sure Tensor rank is not 0
	if err != nil {
		if EnvConfig.Interactive {
			panic(err)
		} else {
			t.Err = err
			return t
		}
	}

	err = verifyGoodInterval(start, end, [2]int{0, t.Size(0)})
	if err != nil {
		if EnvConfig.Interactive {
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
		data:   t.data,
		shape:  shape,
		stride: slices.Clone(t.stride),
		offset: t.offset + start*t.stride[0],
	}
}

// Broadcast broadcasts the Tensor to the given shape.
func (t Tensor[T]) Broadcast(shape ...int) Tensor[T] {
	if t.Err != nil {
		if EnvConfig.Interactive {
			panic(t.Err)
		} else {
			return t
		}
	}

	if !t.Broadable(shape...) {
		if EnvConfig.Interactive {
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
						dstIdx := i*stride + j*shape[axis] + k*newStride[axis]
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
		shape: slices.Clone(shape),
		stride: newStride,
	}
}

// Reverse reverses the order of the elements of the Tensor.
func (t Tensor[T]) Reverse() Tensor[T] {
	if t.Err != nil {
		if EnvConfig.Interactive {
			panic(t.Err)
		} else {
			return t
		}
	}

	for i, j := 0, t.Numel()-1; i < j; i, j = i+1, j-1 {
		t.data[t.offset+i], t.data[t.offset+j] = t.data[t.offset+j], t.data[t.offset+i]
	}

	return t
}

// Flip reverses the order of the elements of the Tensor
// along the given axis.
func (t Tensor[T]) Flip(axis int) Tensor[T] {
	if t.Err != nil {
		if EnvConfig.Interactive {
			panic(t.Err)
		} else {
			return t
		}
	}

	err := verifyAxisBounds(0, len(t.shape))
	if err != nil {
		if EnvConfig.Interactive {
			panic(err)
		} else {
			t.Err = err
			return t
		}
	}

	stride := t.stride[axis]
	for i := 0; i < t.Numel(); i += t.shape[axis] * stride {
		for j, k := 0, t.shape[axis]-1; j < k; j, k = j+1, k-1 {
			for l := 0; l < t.stride[axis]; l++ {
				t.data[t.offset+i+j*stride+l], t.data[t.offset+i+k*stride+l] = t.data[t.offset+i+k*stride+l], t.data[t.offset+i+j*stride+l]
			}
		}
	}

	return t
}

// Repeat repeats the elements of the array n times.
func (t Tensor[T]) Repeat(n int) Tensor[T] {
	if t.Err != nil {
		if EnvConfig.Interactive {
			panic(t.Err)
		} else {
			return t
		}
	}

	numel := t.Numel()
	dataBuf := t.Ravel()
	data := slices.WithLen[T](n * numel)
	for i := 0; i < n; i++ {
		copy(data[i*numel:i*numel+numel], dataBuf)
	}

	shape := slices.WithLen[int](len(t.shape) + 1)
	stride := slices.WithLen[int](len(t.stride) + 1)

	shape[0] = n
	copy(shape[1:], t.shape)
	stride[0] = t.shape[0] * t.stride[0]
	copy(stride[1:], t.stride)

	return Tensor[T]{
		data: data,
		shape: shape,
		stride: stride,
	}
}

// Permute permutes the Tensor's axes without changing the data.
func (t Tensor[T]) Permute(axes ...int) Tensor[T] {
	if t.Err != nil {
		if EnvConfig.Interactive {
			panic(t.Err)
		} else {
			return t
		}
	}

	err := verifyArgsBounds(len(axes), len(t.shape))
	if err != nil {
		if EnvConfig.Interactive {
			panic(err)
		} else {
			t.Err = err
			return t
		}
	}

	shapeCopy := slices.Clone(t.shape)
	strideCopy := slices.Clone(t.stride)

	newshape := slices.WithLen[int](len(t.shape))
	newstride := slices.WithLen[int](len(t.stride))

	for i, axis := range axes {
		err := verifyAxisBounds(axis, len(t.shape))
		if err != nil {
			if EnvConfig.Interactive {
				panic(err)
			} else {
				t.Err = err
				return t
			}
		}

		newshape[i] = shapeCopy[axis]
		newstride[i] = strideCopy[axis]
	}

	return Tensor[T]{
		data: t.data,
		shape: newshape,
		stride: newstride,
		offset: t.offset,
	}
}

// Cat concatenates the other Tensor to this Tensor along the given axis.
func (t Tensor[T]) Cat(other Tensor[T], axis int) Tensor[T] {
	if t.Err != nil {
		if EnvConfig.Interactive {
			panic(t.Err)
		} else {
			return t
		}
	}

	if other.Err != nil {
		if EnvConfig.Interactive {
			panic("nune: could not concatenate the two tensors")
		} else {
			t.Err = errors.New("nune: could not concatenate the two tensors")
			return t
		}
	}

	err := verifyAxisBounds(axis, len(t.shape))
	if err != nil {
		if EnvConfig.Interactive {
			panic(err)
		} else {
			t.Err = err
			return t
		}
	}

	if !slices.Equal(t.shape[:axis], other.shape[:axis]) || !slices.Equal(t.shape[axis+1:], other.shape[axis+1:]) {
		if EnvConfig.Interactive {
			panic("nune: tensors' shapes do not allow concatenating them")
		} else {
			t.Err = errors.New(("nune: tensors' shapes do not allow concatenating them"))
			return t
		}
	}

	newshape := slices.Clone(t.shape)
	newshape[axis] += other.shape[axis]
	newstride := configStride(newshape)

	ts := t.stride[axis]
	os := other.stride[axis]
	ns := newstride[axis]
	data := slices.WithLen[T](t.Numel() + other.Numel())

	// how do I come up with these algorithms...
	for i := 0; i < t.Numel()/(ts*t.shape[axis]); i++ {
		copy(data[i*ns*newshape[axis]:i*ns*newshape[axis]+ts*t.shape[axis]], t.Ravel()[i*ts*t.shape[axis]:(i+1)*ts*t.shape[axis]])
	}

	for i := 0; i < other.Numel()/(os*other.shape[axis]); i++ {
		copy(data[i*ns*newshape[axis]+ts*t.shape[axis]:(i+1)*ns*newshape[axis]], other.Ravel()[i*os*other.shape[axis]:(i+1)*os*other.shape[axis]])
	}

	return Tensor[T]{
		data: data,
		shape: newshape,
		stride: newstride,
	}
}

// Stack stacks this and the other Tensor together along a new axis.
func (t Tensor[T]) Stack(other Tensor[T], axis int) Tensor[T] {
	if t.Err != nil {
		if EnvConfig.Interactive {
			panic(t.Err)
		} else {
			return t
		}
	}

	if other.Err != nil {
		if EnvConfig.Interactive {
			panic("nune: could not concatenate the two tensors")
		} else {
			t.Err = errors.New("nune: could not concatenate the two tensors")
			return t
		}
	}

	err := verifyAxisBounds(axis, len(t.shape))
	if err != nil {
		if EnvConfig.Interactive {
			panic(err)
		} else {
			t.Err = err
			return t
		}
	}

	t = t.Unsqueeze(axis)
	other = other.Unsqueeze(axis)
	t = t.Cat(other, axis)

	return t
}

// Squeeze removes an axis of dimensions 1 from the Tensor's shape.
func (t Tensor[T]) Squeeze(axis int) Tensor[T] {
	if t.Err != nil {
		if EnvConfig.Interactive {
			panic(t.Err)
		} else {
			return t
		}
	}

	err := verifyAxisBounds(axis, len(t.shape))
	if err != nil {
		if EnvConfig.Interactive {
			panic(err)
		} else {
			t.Err = err
			return t
		}
	}

	if t.shape[axis] > 1 {
		if EnvConfig.Interactive {
			panic("nune: tensor axis dimensions greater than 1")
		} else {
			t.Err = errors.New("nune: tensor axis dimensions greater than 1")
			return t
		}
	}

	newshape := slices.WithLen[int](len(t.shape) - 1)
	newstride := slices.WithLen[int](len(t.stride) - 1)

	copy(newshape[:axis], t.shape[:axis])
	copy(newshape[axis:], t.shape[axis+1:])
	copy(newstride[:axis], t.stride[:axis])
	copy(newstride[axis:], t.stride[axis+1:])

	return Tensor[T]{
		data: t.data,
		shape: newshape,
		stride: newstride,
		offset: t.offset,
	}
}

// Unsqueeze adds an axis of dimensions 1 to the Tensor's shape.
func (t Tensor[T]) Unsqueeze(axis int) Tensor[T] {
	if t.Err != nil {
		if EnvConfig.Interactive {
			panic(t.Err)
		} else {
			return t
		}
	}

	err := verifyAxisBounds(axis, len(t.shape))
	if err != nil {
		if EnvConfig.Interactive {
			panic(err)
		} else {
			t.Err = err
			return t
		}
	}

	newshape := slices.WithLen[int](len(t.shape) + 1)
	newstride := slices.WithLen[int](len(t.stride) + 1)

	copy(newshape[:axis], t.shape[:axis])
	copy(newstride[:axis], t.stride[:axis])
	newshape[axis] = 1

	if axis < len(t.shape) {
		copy(newshape[axis+1:], t.shape[axis:])
		copy(newstride[axis+1:], t.stride[axis:])
		newstride[axis] = t.shape[axis] * t.stride[axis]
	} else {
		newstride[axis] = 1
	}

	return Tensor[T]{
		data: t.data,
		shape: newshape,
		stride: newstride,
		offset: t.offset,
	}
}
