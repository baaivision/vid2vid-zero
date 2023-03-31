


<div align="center">

<h1>vid2vid-zero for Zero-Shot Video Editing</h1>

[Wen Wang](https://scholar.google.com/citations?user=1ks0R04AAAAJ&hl=zh-CN)<sup>1*</sup>, &nbsp; [Kangyang Xie](https://github.com/felix-ky)<sup>1*</sup>, &nbsp; [Zide Liu](https://github.com/zideliu)<sup>1*</sup>, &nbsp; [Hao Chen](https://scholar.google.com.au/citations?user=FaOqRpcAAAAJ&hl=en)<sup>1</sup>, &nbsp; [Yue Cao](http://yue-cao.me/)<sup>2</sup>, &nbsp; [Xinlong Wang](https://www.xloong.wang/)<sup>2</sup>, &nbsp; [Chunhua Shen](https://cshen.github.io/)<sup>1</sup>

<sup>1</sup>[ZJU](https://www.zju.edu.cn/english/), &nbsp; <sup>2</sup>[BAAI](https://www.baai.ac.cn/english.html)

<br>
  
<image src="docs/vid2vid-zero.png" />
<br>

</div>

We propose vid2vid-zero, a simple yet effective method for zero-shot video editing. Our vid2vid-zero leverages off-the-shelf image diffusion models, and doesn't require training on any video. At the core of our method is a null-text inversion module for text-to-video alignment, a cross-frame modeling module for temporal consistency, and a spatial regularization module for fidelity to the original video. Without any training, we leverage the dynamic nature of the attention mechanism to enable bi-directional temporal modeling at test time. 
Experiments and analyses show promising results in editing attributes, subjects, places, etc., in real-world videos. 

[[Paper]](https://arxiv.org/pdf/2303.17599.pdf)

## Hightlights

- Video editing with off-the-shelf image diffusion models.

- No training on any video.

- Promising results in editing attributes, subjects, places, etc., in real-world videos.

## Examples
<table class="center">
<tr>
  <td style="text-align:center;"><b>Input Video</b></td>
  <td style="text-align:center;"><b>Output Video</b></td>
  <td style="text-align:center;"><b>Input Video</b></td>
  <td style="text-align:center;"><b>Output Video</b></td>
</tr>

<tr>
  <td width=25% style="text-align:center;color:gray;">"A car is moving on the road"</td>
  <td width=25% style="text-align:center;">"A Porsche car is moving on the desert"</td>
  <td width=25% style="text-align:center;color:gray;">"A car is moving on the road"</td>
  <td width=25% style="text-align:center;">"A jeep car is moving on the snow"</td>
</tr>

<tr>
  <td style colspan="2"><img src="examples/jeep-moving_Porsche.gif"></td>
  <td style colspan="2"><img src="examples/jeep-moving_snow.gif"></td>       
</tr>


<tr>
  <td width=25% style="text-align:center;color:gray;">"A man is running"</td>
  <td width=25% style="text-align:center;">"Stephen Curry is running in Time Square"</td>
  <td width=25% style="text-align:center;color:gray;">"A man is running"</td>
  <td width=25% style="text-align:center;">"A man is running in New York City"</td>
</tr>

<tr>
  <td style colspan="2"><img src="examples/man-running_stephen.gif"></td>
  <td style colspan="2"><img src="examples/man-running_newyork.gif"></td>       
</tr>

<tr>
  <td width=25% style="text-align:center;color:gray;">"A child is riding a bike on the road"</td>
  <td width=25% style="text-align:center;">"a child is riding a bike on the flooded road"</td>
  <td width=25% style="text-align:center;color:gray;">"A child is riding a bike on the road"</td>
  <td width=25% style="text-align:center;">"a lego child is riding a bike on the road.gif"</td>
</tr>

<tr>
  <td style colspan="2"><img src="examples/child-riding_flooded.gif"></td>
  <td style colspan="2"><img src="examples/child-riding_lego.gif"></td>       
</tr>

<tr>
  <td width=25% style="text-align:center;color:gray;">"A car is moving on the road"</td>
  <td width=25% style="text-align:center;">"A car is moving on the snow"</td>
  <td width=25% style="text-align:center;color:gray;">"A car is moving on the road"</td>
  <td width=25% style="text-align:center;">"A jeep car is moving on the desert"</td>
</tr>

<tr>
  <td style colspan="2"><img src="examples/red-moving_snow.gif"></td>
  <td style colspan="2"><img src="examples/red-moving_desert.gif"></td>       
</tr>
</table>


## Citation

```
@article{vid2vid-zero,
  title={Zero-Shot Video Editing Using Off-The-Shelf Image Diffusion Models},
  author={Wang, Wen and Xie, kangyang and Liu, Zide and Chen, Hao and Cao, Yue and Wang, Xinlong and Shen, Chunhua},
  journal={arXiv preprint arXiv:2303.17599},
  year={2023}
}
```

## Acknowledgement
[Tune-A-Video](https://github.com/showlab/Tune-A-Video), [diffusers](https://github.com/huggingface/diffusers).

## Contact

**We are hiring** at all levels at BAAI Vision Team, including full-time researchers, engineers and interns. 
If you are interested in working with us on **foundation model, visual perception and multimodal learning**, please contact [Xinlong Wang](https://www.xloong.wang/) (`wangxinlong@baai.ac.cn`) and [Yue Cao](http://yue-cao.me/) (`caoyue@baai.ac.cn`).
