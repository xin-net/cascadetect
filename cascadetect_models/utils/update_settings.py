from ultralytics import settings

# 显示当前设置
print("当前设置:")
print(settings)

# 更新TensorBoard设置为True
settings.update({'tensorboard': True})

# 显示更新后的设置
print("\n更新后的设置:")
print(settings)

print("\nTensorBoard设置已更新为:", settings.get('tensorboard')) 