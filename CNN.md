These two concepts are fundamental reasons why CNNs work well for images. I’ll explain them clearly with intuition and examples.

1. Locality
Meaning

Locality means that nearby pixels in an image are more related to each other than distant pixels.

In other words, important visual patterns are local.

Example patterns:

edges
corners
textures
small shapes

These patterns usually exist in small neighborhoods of pixels, not across the whole image.

Example

Imagine detecting an eye in a face image.

You don't need the entire image.
You only need a small patch of pixels around the eye.

So instead of connecting every pixel to every neuron, CNNs use small filters like:

3 × 3
5 × 5
7 × 7

These filters capture local patterns.

Why locality is important

Without locality:

A network would treat far-away pixels as equally important
Model would become very large
Training would become inefficient

Locality enables:

fewer parameters
faster training
better generalization
2. Translation Invariance
Meaning

Translation invariance means the model can detect an object even if it appears at different positions in the image.

Example:

A cat could appear:

left side
center
right side
top corner

But it is still the same cat.

A good vision model should recognize it regardless of its location.

How CNN achieves this

CNNs use shared filters.

A filter that detects an edge is applied across the entire image.

Example:

Filter detects vertical edge

That same filter slides across the image:

| image |
 ↓
[ convolution ]
 ↓
feature map

So the same feature detector works everywhere.

Role of pooling

Pooling layers further improve translation invariance.

Example:

MaxPool(2×2)

If an object moves slightly in the image, the pooled feature still remains strong.

Locality vs Translation Invariance (Quick Comparison)
Concept	Meaning
Locality	Important patterns exist in small regions
Translation Invariance	Objects can be recognized anywhere in the image
Simple Visual Intuition
Locality
Image

[ small region ]
   ↑
edge detector

Focus on small neighborhood.

Translation invariance
Object at different positions

Cat here → detected
Cat here → detected
Cat here → detected

Same filter works everywhere.

Why CNNs are better than Fully Connected Networks for images

Fully connected network:

every pixel connected to every neuron

Problems:

millions of parameters
no spatial structure
poor generalization

CNN uses:

local receptive fields
+
shared filters

Which gives:

locality
translation invariance

If you'd like, I can also explain three more concepts that interviewers almost always ask with locality and translation invariance:

Receptive Field
Equivariance vs Invariance
Why CNNs fail at rotation invariance

These are very common deep learning interview questions.

yes

Great — these three concepts are almost always asked together with locality and translation invariance in deep learning interviews.

1. Receptive Field
Meaning

The receptive field of a neuron is the region of the input image that influences that neuron’s output.

In CNNs, each neuron does not see the whole image.
It only sees a small part of the image.

Example

Suppose we apply a 3×3 convolution filter.

Input Image

[ pixels ]

A neuron in the feature map only looks at:

3 × 3 pixels

So the receptive field = 3×3 region of the input.

Receptive Field Grows in Deeper Layers

Each layer increases the effective receptive field.

Example:

Layer structure

Conv(3×3)
Conv(3×3)
Conv(3×3)

Effective receptive field:

Layer 1 → 3×3
Layer 2 → 5×5
Layer 3 → 7×7

So deeper neurons see larger portions of the image.

Why this is important

Early layers detect:

edges
corners
textures

Deeper layers detect:

object parts
faces
cars
animals

This hierarchical feature learning is what makes CNNs powerful.

2. Translation Equivariance vs Translation Invariance

This is a very common interview trick question.

CNNs are actually translation equivariant, not perfectly invariant.

Translation Equivariance
Meaning

If the input shifts, the output feature map shifts in the same way.

Example:

Input image shifts right

Then

Feature map also shifts right

Mathematically

f(T(x)) = T(f(x))

Where

T = translation
Example

If a cat moves right in the image:

Input
[ cat     ]

Feature map
[ feature ]

Move cat right:

Input
[     cat ]

Feature map
[     feature ]

Feature map shifts accordingly.

Translation Invariance
Meaning

The output prediction stays the same even if the object moves.

Example:

Cat left → predicted cat
Cat right → predicted cat

So the classification result is unchanged.

How CNNs Achieve Translation Invariance

Through:

Pooling layers
Global average pooling
Fully connected layers

Pooling example:

MaxPool(2×2)

Small movements in objects do not change pooled outputs much.

Key Interview Answer

If asked:

“Are CNNs translation invariant?”

Correct answer:

CNNs are translation equivariant by design, but pooling and global aggregation make them approximately translation invariant.

3. Why CNNs Fail at Rotation Invariance

CNNs struggle when objects rotate.

Example:

Digit 6

Rotate it:

Digit 9

The CNN may misclassify.

Why this happens

CNN filters are learned for specific orientations.

Example filter:

detect vertical edge

But if the object rotates:

edge becomes diagonal

The filter no longer activates strongly.

Example

Train CNN on upright cats:

🐱

Test on rotated cat:

↺ 🐱

CNN might fail.

Solutions

Researchers use several methods:

1. Data Augmentation

Rotate images during training.

0°
45°
90°
180°
2. Spatial Transformer Networks

Networks learn to normalize orientation.

Example architecture:

Spatial Transformer Networks
3. Group Equivariant CNNs

These CNNs learn filters that rotate.

Example concept:

Group Equivariant Convolution
Summary
Concept	Meaning
Locality	CNN filters focus on small local regions
Receptive Field	Portion of input affecting a neuron
Translation Equivariance	Output shifts when input shifts
Translation Invariance	Prediction unchanged by object movement
Rotation Problem	CNNs struggle with rotated objects

💡 Since you're preparing for vision / perception interviews, the next three topics interviewers almost always ask after this are:

Feature Hierarchy in CNNs (edges → textures → objects)
Why pooling works mathematically
Difference between stride, padding, and dilation

If you'd like, I can also explain the complete intuition of CNNs from first principles (this is one of the strongest answers in interviews).