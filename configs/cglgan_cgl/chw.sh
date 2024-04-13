DEBUG=False
ADDITIONAL_ARGS=" \
++optimizer.weight_decay=1e-4 \
++generator.auxilary_task=chw \
++generator.in_channels=10 \
++discriminator.in_channels=10 \
"
EXP_ID="chw"