<template>
  <div class="image-display-panel panel">

    <div class="image-container" ref="imageContainerRef" @mousedown="handleMouseDown" @mousemove="handleMouseMove" @mouseup="handleMouseUp" @mouseleave="handleMouseLeave">
      <div class="zoom-controls">
        <button @click="zoomIn">放大</button>
        <button @click="zoomOut">缩小</button>
        <button @click="fitToScreen">适应屏幕</button>
      </div>
      <img v-if="imageUrl" :src="imageUrlLocal" alt="缺陷图片" ref="imageRef" @load="onImageLoad" />
      <p v-if="!imageUrl">请从左侧列表选择一张图片</p>
      <canvas ref="canvasRef"></canvas>
    </div>
  </div>
</template>

<script setup>
import { ref, watch, onMounted, onUnmounted, nextTick } from 'vue';

const props = defineProps({
  imageUrl: String,
  annotations: Array,
});

const emit = defineEmits(['update-annotation', 'create-annotation', 'delete-annotation']);

const imageRef = ref(null);
const canvasRef = ref(null);
const imageContainerRef = ref(null);
const imageUrlLocal = ref(''); // 用于确保图片重新加载

let ctx = null;
let isDrawing = false;
let startX, startY, currentX, currentY;
let currentDrawingAnnotation = null; // 当前正在绘制的标注
let imageScale = ref(1); // 图片在容器中的缩放比例
let imageOffset = ref({ x: 0, y: 0 }); // 图片在容器中的偏移

const zoomIn = () => {
  imageScale.value *= 1.2;
  applyImagePosition();
};

const zoomOut = () => {
  imageScale.value /= 1.2;
  // 限制最小缩放比例，防止图片过小
  if (imageScale.value < 0.1) imageScale.value = 0.1;
  applyImagePosition();
};

const fitToScreen = () => {
  if (!imageRef.value || !imageContainerRef.value) return;
  const img = imageRef.value;
  const container = imageContainerRef.value;
  const containerWidth = container.clientWidth;
  const containerHeight = container.clientHeight;
  const scaleX = containerWidth / img.naturalWidth;
  const scaleY = containerHeight / img.naturalHeight;
  imageScale.value = Math.min(scaleX, scaleY);
  applyImagePosition();
};

const applyImagePosition = () => {
  if (!imageRef.value || !canvasRef.value || !imageContainerRef.value) return;

  const img = imageRef.value;
  const container = imageContainerRef.value;
  const canvas = canvasRef.value;

  const containerWidth = container.clientWidth;
  const containerHeight = container.clientHeight;

  const actualImageWidth = img.naturalWidth * imageScale.value;
  const actualImageHeight = img.naturalHeight * imageScale.value;

  // 确保图片居中显示
  imageOffset.value.x = (containerWidth - actualImageWidth) / 2;
  imageOffset.value.y = (containerHeight - actualImageHeight) / 2;

  // 更新图片和canvas的尺寸和位置
  img.style.width = `${actualImageWidth}px`;
  img.style.height = `${actualImageHeight}px`;
  img.style.left = `${imageOffset.value.x}px`;
  img.style.top = `${imageOffset.value.y}px`;

  canvas.width = actualImageWidth;
  canvas.height = actualImageHeight;
  canvas.style.left = `${imageOffset.value.x}px`;
  canvas.style.top = `${imageOffset.value.y}px`;

  drawAnnotations();
};

const onImageLoad = () => {
  if (imageRef.value && canvasRef.value && imageContainerRef.value) {
    const img = imageRef.value;
    const container = imageContainerRef.value;
    const canvas = canvasRef.value;

    // 计算图片在容器内的缩放比例和偏移，以使其完整显示并居中
    const containerWidth = container.clientWidth;
    const containerHeight = container.clientHeight;
    const imgAspectRatio = img.naturalWidth / img.naturalHeight;
    const containerAspectRatio = containerWidth / containerHeight;

    // 设置canvas尺寸与图片一致
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    // 将canvas定位到图片上
    canvas.style.position = 'absolute';

    // 计算图片在容器中的缩放比例，用于标注转换
    // 假设图片会通过 object-fit: contain 填充容器
    const scaleX = containerWidth / img.naturalWidth;
    const scaleY = containerHeight / img.naturalHeight;
    imageScale.value = Math.min(scaleX, scaleY);

    // 计算图片实际显示尺寸和偏移，用于标注位置调整
    const actualImageWidth = img.naturalWidth * imageScale.value;
    const actualImageHeight = img.naturalHeight * imageScale.value;
    imageOffset.value.x = (containerWidth - actualImageWidth) / 2;
    imageOffset.value.y = (containerHeight - actualImageHeight) / 2;

    ctx = canvas.getContext('2d');
    fitToScreen(); // 初始加载时适应屏幕
  }
};

const drawAnnotations = () => {
  if (!ctx || !imageRef.value) return;

  const canvas = canvasRef.value;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // 绘制已有的标注
  props.annotations.forEach(ann => {
    if (ann.x1 === undefined) return; // 跳过无效标注
    const defectColors = {
      '气孔': 'red',
      '气泡': 'blue',
      '水纹': 'green',
      // 可以根据需要添加更多缺陷类型和颜色
      '新缺陷': 'purple' // 用户手动创建的标注
    };
    const color = defectColors[ann.defect_type] || 'orange'; // 默认为橙色

    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    // 将原始坐标按缩放比例转换到canvas坐标，并考虑图片偏移
    const x1 = ann.x1 * imageScale.value;
    const y1 = ann.y1 * imageScale.value;
    const width = (ann.x2 - ann.x1) * imageScale.value;
    const height = (ann.y2 - ann.y1) * imageScale.value;
    ctx.strokeRect(x1, y1, width, height);

    // 显示缺陷类型和置信度
    ctx.fillStyle = color;
    ctx.font = '14px Arial';
    ctx.fillText(`${ann.defect_type} (${ann.confidence.toFixed(2)})`, x1, y1 > 15 ? y1 - 5 : y1 + 15);
  });

  // 如果正在绘制新标注，也绘制它
  if (isDrawing && currentDrawingAnnotation) {
    ctx.strokeStyle = 'blue'; // 新标注用蓝色
    ctx.lineWidth = 2;
    const { x1, y1, x2, y2 } = currentDrawingAnnotation;
    // 坐标已经是相对于canvas的，不需要再乘以scale
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
  }
};

const handleMouseDown = (event) => {
  if (!imageRef.value || !ctx) return;
  // 检查是否点击在图片区域内
  const rect = canvasRef.value.getBoundingClientRect();
  const clickX = event.clientX - rect.left;
  const clickY = event.clientY - rect.top;

  // 将点击坐标转换为相对于图片内容的坐标
  const imgClickX = (clickX - imageOffset.value.x) / imageScale.value;
  const imgClickY = (clickY - imageOffset.value.y) / imageScale.value;

  // 检查是否点击在图片区域内（考虑偏移和缩放后的实际图片区域）
  if (imgClickX < 0 || imgClickX > imageRef.value.naturalWidth || imgClickY < 0 || imgClickY > imageRef.value.naturalHeight) {
      return;
  }

  isDrawing = true;
  startX = clickX;
  startY = clickY;
  currentDrawingAnnotation = { x1: startX, y1: startY, x2: startX, y2: startY };
};

const handleMouseMove = (event) => {
  if (!isDrawing || !imageRef.value || !ctx) return;
  const rect = canvasRef.value.getBoundingClientRect();
  currentX = event.clientX - rect.left;
  currentY = event.clientY - rect.top;

  // 限制在图片实际显示区域内
  const actualImageWidth = imageRef.value.naturalWidth * imageScale.value;
  const actualImageHeight = imageRef.value.naturalHeight * imageScale.value;

  currentX = Math.max(imageOffset.value.x, Math.min(currentX, imageOffset.value.x + actualImageWidth));
  currentY = Math.max(imageOffset.value.y, Math.min(currentY, imageOffset.value.y + actualImageHeight));

  currentDrawingAnnotation.x2 = currentX;
  currentDrawingAnnotation.y2 = currentY;
  drawAnnotations(); // 实时重绘
};

const handleMouseUp = () => {
  if (!isDrawing || !imageRef.value || !ctx) return;
  isDrawing = false;
  if (currentDrawingAnnotation) {
    // 确保 x1 < x2 and y1 < y2
    const x1 = Math.min(currentDrawingAnnotation.x1, currentDrawingAnnotation.x2);
    const y1 = Math.min(currentDrawingAnnotation.y1, currentDrawingAnnotation.y2);
    const x2 = Math.max(currentDrawingAnnotation.x1, currentDrawingAnnotation.x2);
    const y2 = Math.max(currentDrawingAnnotation.y1, currentDrawingAnnotation.y2);

    // 忽略非常小的框
    if (Math.abs(x2 - x1) < 5 || Math.abs(y2 - y1) < 5) {
        currentDrawingAnnotation = null;
        drawAnnotations(); // 清除可能画的小点
        return;
    }

    // 将canvas坐标转换回原始图片坐标
    const newAnnotation = {
      x1: (x1 - imageOffset.value.x) / imageScale.value,
      y1: (y1 - imageOffset.value.y) / imageScale.value,
      x2: (x2 - imageOffset.value.x) / imageScale.value,
      y2: (y2 - imageOffset.value.y) / imageScale.value,
      defect_type: '新缺陷', // 默认类型，可由用户在右侧面板修改
      confidence: 1.0, // 用户手动创建，置信度默认为1
    };
    emit('create-annotation', newAnnotation);
    currentDrawingAnnotation = null; // 重置
    // drawAnnotations(); // 创建后会由父组件刷新，这里可以不画
  }
};

const handleMouseLeave = () => {
    // 如果鼠标移出时仍在绘制，则取消绘制
    if (isDrawing) {
        isDrawing = false;
        currentDrawingAnnotation = null;
        drawAnnotations(); // 清除未完成的框
    }
};

watch(() => props.imageUrl, (newUrl) => {
  imageUrlLocal.value = newUrl ? `${newUrl}?t=${new Date().getTime()}` : ''; // 添加时间戳确保图片刷新
  // 图片URL变化时，等待图片加载完成后再绘制
  if (newUrl) {
    nextTick(() => {
      // onImageLoad 会在图片加载后被调用
    });
  }else {
      if(ctx && canvasRef.value) {
          ctx.clearRect(0, 0, canvasRef.value.width, canvasRef.value.height);
      }
  }
});

watch(() => props.annotations, () => {
  if (imageRef.value && imageRef.value.complete) { // 确保图片已加载
      onImageLoad(); // 图片已加载，直接重绘
  } else if (imageRef.value) {
      // 图片未加载完成，等待加载事件触发 onImageLoad
  }
}, { deep: true });

onMounted(() => {
  if (canvasRef.value) {
    ctx = canvasRef.value.getContext('2d');
  }
  // 如果初始就有图片URL，尝试加载
  if (props.imageUrl) {
      imageUrlLocal.value = `${props.imageUrl}?t=${new Date().getTime()}`;
  }
  window.addEventListener('keydown', handleKeyDown);
});

onUnmounted(() => {
  window.removeEventListener('keydown', handleKeyDown);
});

const handleKeyDown = (event) => {
  if (!imageRef.value || imageScale.value <= 1) return; // 只在放大时允许平移

  const step = 20; // 平移步长
  let moved = false;

  switch (event.key) {
    case 'ArrowUp':
      imageOffset.value.y += step;
      moved = true;
      break;
    case 'ArrowDown':
      imageOffset.value.y -= step;
      moved = true;
      break;
    case 'ArrowLeft':
      imageOffset.value.x += step;
      moved = true;
      break;
    case 'ArrowRight':
      imageOffset.value.x -= step;
      moved = true;
      break;
  }

  if (moved) {
    // 限制图片偏移量，确保图片不会移出容器可视范围
    const img = imageRef.value;
    const container = imageContainerRef.value;
    const actualImageWidth = img.naturalWidth * imageScale.value;
    const actualImageHeight = img.naturalHeight * imageScale.value;

    const containerWidth = container.clientWidth;
    const containerHeight = container.clientHeight;

    // 计算最大允许的偏移量（图片边缘与容器边缘对齐）
    const maxX = (actualImageWidth > containerWidth) ? (actualImageWidth - containerWidth) / 2 : 0;
    const maxY = (actualImageHeight > containerHeight) ? (actualImageHeight - containerHeight) / 2 : 0;

    // 限制 x 轴偏移
    imageOffset.value.x = Math.max(imageOffset.value.x, containerWidth - actualImageWidth + maxX);
    imageOffset.value.x = Math.min(imageOffset.value.x, maxX);

    // 限制 y 轴偏移
    imageOffset.value.y = Math.max(imageOffset.value.y, containerHeight - actualImageHeight + maxY);
    imageOffset.value.y = Math.min(imageOffset.value.y, maxY);

    applyImagePosition();
    event.preventDefault(); // 阻止默认的滚动行为
  }
};

</script>

<style scoped>
.image-display-panel {
  width: 60%; /* 屏幕3/5宽度 */
  flex-grow: 0; /* 不再占据剩余空间，而是固定宽度 */
  height: 100%;
  display: flex;
  flex-direction: column; /* 垂直布局 */
  overflow: hidden; /* 确保图片和canvas不会超出面板 */
  position: relative; /* 为canvas绝对定位提供基准 */
  background-color: #e9ecef; /* 面板背景色 */
}



.image-container {
  position: relative;
  width: 100%;
  height: 100%; /* 图片容器占据剩余高度 */
  display: flex;
  justify-content: center;
  align-items: center;
  background-color: #e0e0e0;
  overflow: hidden; /* 隐藏超出部分 */
}

.image-container img {
  object-fit: contain; /* 保持图片比例并完整显示 */
  display: block; /* 移除图片下方的额外空间 */
}

.image-container canvas {
  pointer-events: none; /* 确保canvas不会捕获鼠标事件，除非在特定交互时启用 */
}

.image-container p {
  color: #666;
}

.zoom-controls {
  position: absolute;
  top: 10px;
  left: 10px;
  z-index: 10; /* 确保按钮在图片和canvas之上 */
  display: flex;
  gap: 5px;
}

.zoom-controls button {
  padding: 5px 10px;
  background-color: rgba(0, 0, 0, 0.5);
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
}

.zoom-controls button:hover {
  background-color: rgba(0, 0, 0, 0.7);
}
</style>