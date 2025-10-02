# Differences from the HypCD code that might improve results

HypCD uses Dyno with registers. I don't use registers

SupConLoss does not change the direction of the similarity metrics even for distance. I have no idea how they can get away with this
 
HypCD combines distance and angle loss first. I combine supervised and self-supervised loss first.

It seems like they do not do hyperbolic training all the time? There is a term called hyper_start_epoch and hyper_end_epoch. However, it seems like it starts at 0 and ends at 200, so it is not really used.

For both the angle starts at max_weight and moves to 0 as time progresses. Max_weight being 1

HypCD has absolutely no gradient modifications. At least in train script
HypCD uses a projection head that is missing the last layer, which goes from bottleneck dim to out dim. However, this is fine, since the bottleneck dim can be used as out_dim, as the hidden_dim still exists for hidden layers. 