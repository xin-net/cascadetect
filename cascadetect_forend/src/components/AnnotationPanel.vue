<template>
  <div class="annotation-panel panel">
    <h3>标注信息</h3>
    <div v-if="!imageId">请先选择一张图片</div>
    <div v-if="imageId && annotationsLocal.length === 0">该图片暂无标注信息，您可以在图片上拖拽创建新的标注。</div>
    <ul v-if="imageId && annotationsLocal.length > 0">
      <li v-for="(ann, index) in annotationsLocal" :key="ann.id || `new-${index}`">
        <div v-if="!ann.isEditing">
          <p><strong>缺陷类型:</strong> {{ ann.defect_type }}</p>
          <p><strong>置信度:</strong> {{ ann.confidence ? ann.confidence.toFixed(2) : 'N/A' }}</p>
          <p><strong>位置 (x1, y1, x2, y2):</strong> {{ ann.x1.toFixed(1) }}, {{ ann.y1.toFixed(1) }}, {{ ann.x2.toFixed(1) }}, {{ ann.y2.toFixed(1) }}</p>
          <div class="annotation-actions">
            <button @click="editAnnotation(ann)">编辑</button>
            <button @click="confirmDelete(ann.id)" class="delete-btn">删除</button>
          </div>
        </div>
        <div v-else class="edit-form">
          <label>缺陷类型:</label>
          <select v-model="ann.defect_type">
            <option v-for="type in defectTypes" :key="type" :value="type">{{ type }}</option>
          </select>
          <label>置信度: (只读)</label>
          <input type="number" v-model.number="ann.confidence" readonly />
          <label>x1:</label>
          <input type="number" v-model.number="ann.x1" @input="validateCoordinates(ann)"/>
          <label>y1:</label>
          <input type="number" v-model.number="ann.y1" @input="validateCoordinates(ann)"/>
          <label>x2:</label>
          <input type="number" v-model.number="ann.x2" @input="validateCoordinates(ann)"/>
          <label>y2:</label>
          <input type="number" v-model.number="ann.y2" @input="validateCoordinates(ann)"/>
          <button @click="saveAnnotation(ann)">保存</button>
          <button @click="cancelEdit(ann)">取消</button>
        </div>
      </li>
    </ul>
    <div v-if="imageId" class="add-annotation-section">
        <h4>手动添加新标注</h4>
        <label>缺陷类型:</label>
        <select v-model="newAnnotationForm.defect_type">
            <option v-for="type in defectTypes" :key="type" :value="type">{{ type }}</option>
        </select>
        <label>x1:</label>
        <input type="number" v-model.number="newAnnotationForm.x1" placeholder="例如: 100.5"/>
        <label>y1:</label>
        <input type="number" v-model.number="newAnnotationForm.y1" placeholder="例如: 150.2"/>
        <label>x2:</label>
        <input type="number" v-model.number="newAnnotationForm.x2" placeholder="例如: 200.0"/>
        <label>y2:</label>
        <input type="number" v-model.number="newAnnotationForm.y2" placeholder="例如: 250.8"/>
        <button @click="submitNewAnnotation">添加标注</button>
        <p v-if="newAnnotationError" class="error-message">{{ newAnnotationError }}</p>
    </div>
  </div>
</template>

<script setup>
import { ref, watch, reactive } from 'vue';

const props = defineProps({
  annotations: Array,
  imageId: [Number, String, null]
});

const emit = defineEmits(['update-annotation', 'delete-annotation', 'create-annotation']);

const annotationsLocal = ref([]);
const defectTypes = ref(['正常', '气孔', '裂纹', '夹渣', '其他']); // 与后端/模型保持一致
const newAnnotationError = ref('');

const newAnnotationForm = reactive({
    defect_type: defectTypes.value[0], // 默认选择第一个
    x1: null,
    y1: null,
    x2: null,
    y2: null,
    confidence: 1.0 // 手动添加，置信度默认为1
});

watch(() => props.annotations, (newAnnotations) => {
  // 深拷贝，并为每个标注添加isEditing状态
  annotationsLocal.value = newAnnotations ? JSON.parse(JSON.stringify(newAnnotations)).map(ann => ({ ...ann, isEditing: false, originalState: null })) : [];
}, { deep: true, immediate: true });

const editAnnotation = (annotation) => {
  // 保存原始状态以便取消
  annotation.originalState = JSON.parse(JSON.stringify(annotation));
  annotation.isEditing = true;
};

const saveAnnotation = (annotation) => {
  if (!validateCoordinates(annotation, true)) return;
  const { originalState, ...annToSave } = annotation; // 移除originalState
  emit('update-annotation', annToSave);
  annotation.isEditing = false;
  annotation.originalState = null; // 清除原始状态
};

const cancelEdit = (annotation) => {
  // 恢复原始状态
  if (annotation.originalState) {
    Object.assign(annotation, annotation.originalState);
  }
  annotation.isEditing = false;
  annotation.originalState = null;
};

const confirmDelete = (annotationId) => {
  if (window.confirm('确定要删除此标注吗？')) {
    emit('delete-annotation', annotationId);
  }
};

const validateCoordinates = (annotation, showError = false) => {
    const { x1, y1, x2, y2 } = annotation;
    if (x1 == null || y1 == null || x2 == null || y2 == null) {
        if (showError) alert('坐标值不能为空');
        return false;
    }
    if (x1 >= x2 || y1 >= y2) {
        if (showError) alert('坐标格式错误：x1必须小于x2，y1必须小于y2。');
        return false;
    }
    return true;
};

const submitNewAnnotation = () => {
    newAnnotationError.value = '';
    if (newAnnotationForm.x1 == null || newAnnotationForm.y1 == null || newAnnotationForm.x2 == null || newAnnotationForm.y2 == null) {
        newAnnotationError.value = '所有坐标值均不能为空。';
        return;
    }
    if (newAnnotationForm.x1 >= newAnnotationForm.x2 || newAnnotationForm.y1 >= newAnnotationForm.y2) {
        newAnnotationError.value = '坐标格式错误：x1必须小于x2，y1必须小于y2。';
        return;
    }
    if (!props.imageId) {
        newAnnotationError.value = '没有选中的图片，无法添加标注。';
        return;
    }

    emit('create-annotation', { ...newAnnotationForm });
    // 清空表单
    newAnnotationForm.defect_type = defectTypes.value[0];
    newAnnotationForm.x1 = null;
    newAnnotationForm.y1 = null;
    newAnnotationForm.x2 = null;
    newAnnotationForm.y2 = null;
    // confidence 保持为1.0
};

</script>

<style scoped>
.annotation-panel {
  width: 20%; /* 屏幕1/5宽度 */
  height: 100%;
  overflow-y: auto;
  padding: 15px;
  border-left: 1px solid #ddd;
  box-sizing: border-box;
}

h3 {
  text-align: center;
  color: #333;
  margin-bottom: 20px;
}

.annotation-actions {
  display: flex;
  justify-content: space-between;
  gap: 10px; /* 按钮之间的间距 */
}

.annotation-actions button {
  flex: 1; /* 每个按钮占据可用空间的50% */
  margin-right: 0; /* 移除默认的右边距 */
}

ul {
  list-style-type: none;
  padding: 0;
}

li {
  background-color: #f9f9f9;
  border: 1px solid #eee;
  padding: 10px;
  margin-bottom: 10px;
  border-radius: 4px;
}

li p {
  margin: 5px 0;
  font-size: 0.9em;
}

.edit-form label {
  display: block;
  margin-top: 8px;
  font-weight: bold;
  font-size: 0.85em;
}

.edit-form input[type="number"],
.edit-form select {
  width: calc(100% - 16px);
  padding: 6px;
  margin-bottom: 8px;
  border: 1px solid #ccc;
  border-radius: 3px;
  font-size: 0.9em;
}

button {
  margin-right: 5px;
  padding: 6px 10px;
  font-size: 0.9em;
}

.delete-btn {
    background-color: #dc3545; /* 红色删除按钮 */
}
.delete-btn:hover {
    background-color: #c82333;
}

.add-annotation-section {
    margin-top: 20px;
    padding-top: 15px;
    border-top: 1px solid #eee;
}

.add-annotation-section h4 {
    margin-bottom: 10px;
}

.add-annotation-section label {
    display: block;
    margin-top: 8px;
    font-weight: bold;
    font-size: 0.85em;
}

.add-annotation-section input[type="number"],
.add-annotation-section select {
    width: calc(100% - 16px);
    padding: 6px;
    margin-bottom: 8px;
    border: 1px solid #ccc;
    border-radius: 3px;
    font-size: 0.9em;
}

.error-message {
    color: red;
    font-size: 0.9em;
    margin-top: 5px;
}
</style>