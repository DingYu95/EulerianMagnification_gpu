## Implementation notes:

### Inplace vs temporary operations:
Based on my test, in cpu version, inplace modification like `add(a,b,a)` is much faster than `add(a,b,c)`. \
But in gpu version, inplace modification's effect is not as distinguishable as cpu version.
Add `Stream != Null()` also speed up GPU. But pre allocation has little effect.

### Temporal domain DFT for video:
In order to implement ideal frequency chooser, images should be re-arranged in a way that FFT is performed in the time axis, if consider video as 3D volume, a simple solution would be reshape every image into a column, then do horizontal stack them, since `cv::DFT`  only support calculation in **row** axis. However concatenate or insert as column is costly as stated below.

It's known that [OpenCV manages memory](https://docs.opencv.org/2.4/doc/tutorials/core/mat_the_basic_image_container/mat_the_basic_image_container.html) by first continuous allocation for each row and inside each row, different channels for single pixel are store continuously. Thus copy as column will make huge jumps for every pixel. Perhaps some performance can be gained from `cv::split` channel at first place, processing in threads concurrently, `cv::merge` them at final step.

Another issue for *FFT-Based* temporal filters is that with limited memory and to processing in real time, can only do *FFT* in a time window. Image stack becomes a *queue*. However, this cannot be solved by tricks like *head, tail* pointers.

To avoid data copy every frame, I first use modulo to wrap index, and then exploit a shifting property of *FFT*: \
$F\{f(t-t_o)\} = cos(\omega t_o) F(w)$\
generate a Mat of $cos$ values, and do per-element multiplication. Actually, in practice, if the window size is large enough to capture periodic signals like human pulse, we might be able to just ignore this, and do the cycle queue stuff.

### Notes for using `cv::gpu`:

For most PCs, images in RAM are always continuous, however, it's not true for GPU memory. But each row for GpuMat is always continuous. For function like `gpu::reshape`, `gpu::transpose` require `GpuMat` to be continuous. In addition, `GpuMat` better be constructed with value, otherwise there are filled with **nan**, at least in my test.

The memory pattern and DFT in OpenCV cause some trouble for GPU version implementation. Based on my test, it's much more efficient to reshape and stack image as row vertically. Transpose before `gpu::DFT`, then do filtering, and transpose back, retrieve image from rows. Instead of reshape as columns and concatenate.

Another trouble is that `.row()` and `.col()` for `GpuMat` cannot be used as r-value. In order to do assignment correctly, explicit references are needed:
> `gpu::GpuMat curRow = rowImg.row(curFrameIdx);` \
  `Img.copyTo(curRow);`

Instead of `Img.copyTo(rowImg.row(curFrameIdx))`
