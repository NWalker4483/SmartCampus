# APL Smart Campus Multi-Camera Tracking Update #1 
#### By: Nile Walker on 10/1/21
Where did I come from?
Establishing regions of identity transfer across static multi camera systems

## Objective Recap:
* Given footage from multiple views within a scene, identify individuals traveling through the scene and maintain that identity across the multiple perspectives. Any solution should be able to incorporate new cameras down the line with limited disturbance to the performance of re-identification within cameras present during initial training.

## Assumptions:
* The footage is from a static camera.
* The real world positioning of the cameras is not provided.
* The footage is synchronized such that frame n from any camera will represent roughly the same real world time as frame n from any other camera.
* Labeled multi camera track-lets are extremely limited and may not be available at all.

## Current Problem to Solve:
### **Where did I come from?**
As an instance-level recognition problem, person re-ID faces two major challenges. First, the intra-class (instance/identity) variations are typically big due to the changes of camera viewing conditions. For instance, the view change across cameras (frontal to back) brings large appearance changes, making matching the same person difficult. Second, there are also small inter-class variations – people in public space often wear similar clothing; from a distance as typically in surveillance videos,they can look incredibly similar.

Given this it would be extremely useful to if we could leverage the fact that we're observing a realtime surveillance system and there are several constraints that can be placed on which track-lets can be linked.
* Identities observed at the same time in the same camera cannot be the same person.
* Identities observed at significantly different times or with a large gap in between them are very unlikely to be the same person.
* Cameras are often placed along passageways and common areas in such a way that the particular sequence of cameras that an individual passes through is predictable

<!-- NOTE TO REWRITE: -->

While the first two are fairly trivial to apply the last one requires that we have some information on the real world placement/relationship of each of the cameras which according to our assumptions we don't have. And without some multi camera track-lets to train from we can't just observe which cameras people tend to reappear in. So in-order to get around this we need some unsupervised method to quantify which camera X identities from camera Y are likely to reappear in.

<!-- ### **Where am I?**
Related to the previous problem in order to have a model that can predict which camera an identity is likely to reappear in we need some camera representation that can be understood and output by the model. A one-hot encoding is initially attractive as its intuitive and discrete but for reasons I'll go into later it throws away very useful information and doesn't expand well with new cameras. -->

## Current Solutions in Code:
### **Where am I and where did I come from?**
My approach attempts to solve these two problems together, and leverages the regularity and large amount of data available in the surveillance space to correlate cameras.

As a pre-processing step a deep-sort model is run over all available video to generate running track-lets containing a camera ID, a pseudo-ID from deep-sort, a list of bounding boxes, and the corresponding frames on which the deductions were made. 
 
 
 by reframing the Multi camera tracking problem into a graft effusion problem we're able to  sidestep the issue of making a continuous and discrete prediction. I buy breaking each image into patches and

  the graph creation has destroyed explicit information about which patches of Come from nearby sections of one camera is here so in order to re-create this information
<img src="" alt="Association Plot"/>

We perform learning in three discrete stages we learn which cameras are related to one another then we learn which areas in a particular camera are related to

By observing the time difference between the start and stop of track across cameras were able to build a probability density function of  are likely an identity is to 

allows us to independently learn spatial temporal features of cameras and then preform image matching to recover identity. By generating track-lets of detections that occur at regular intervals we're able to convert that re identification problem into a sentence completion problem. 


for each camera in the dataset:
    generate a list of track-lets consisting of a positional encoding of each detection the camera
        * Each track-let should contain a bounding box for every N seconds
        * If a detection is not available on a particular time-step linearly interpolate from the most recent detection
        * After no detections are found for a period of time N append an end of life token to the track-let 

for each camera in the dataset:
    for each track-let:
        All available track-lets from other cameras that start within a certain time threshold N of both the start and stop of (A) or exist at the same time as (A) are considered as candidate matches

<!-- As a pre-training step we need to generate camera descriptor vectors -->


After processing each camera and generating our start-end end-start association vectors
    for each pair of values these probability  density functions are averaged with their corresponding camera

Each candidate match is then converted into a vector using a sequence to vector model. this vector is then scaled by the candidate track-lets Camera Discriminator Vector

In order to get a proposed starting domain for each track-let

our systems also generate a 64 dimensional 

The contributions of this work are thus both the concept of omni-scale feature learning and an ef- fective and efficient implementation of it in OSNet. 

Person Re-Identification (ReID) is a challenging problem in many video analytics and surveillance applications, where a person's identity must be associated across a distributed non-overlapping network of cameras. Video-based person ReID has recently gained much interest because it allows capturing discriminant spatio-temporal information from video clips that is unavailable for image-based ReID. Despite recent advances, deep learning (DL) models for video ReID often fail to leverage this information to improve the robustness of feature representations. In this paper, the motion pattern of a person is explored as an additional cue for ReID. In particular, a flow-guided Mutual Attention network is proposed for fusion of image and optical flow sequences using any 2D-CNN backbone, allowing to encode temporal information along with spatial appearance information. Our Mutual Attention network relies on the joint spatial attention between image and optical flow features maps to activate a common set of salient features across them. In addition to flow-guided attention, we introduce a method to aggregate features from longer input streams for better video sequence-level representation. Our extensive experiments on three challenging video ReID datasets indicate that using the proposed Mutual Attention network allows to improve recognition accuracy considerably with respect to conventional gated-attention networks, and state-of-the-art methods for video-based person ReID.


Person re-identification (re-ID), a fundamental task in distributed multi-camera surveillance, aims to match people appearing in different potentially non-overlapping camera views. 

O