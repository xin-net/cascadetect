<template>
  <div class="image-list-panel panel">
    <h3>图片列表</h3>
    <div class="filter-section">
      <input type="text" v-model="filter.filename" placeholder="按文件名筛选" />
      <select v.model="filter.defectType">
        <option value="">按缺陷类型筛选</option>
        <!-- 假设缺陷类型从后端或配置中获取 -->
        <option v-for="type in defectTypes" :key="type" :value="type">{{ type }}</option>
      </select>
      <input type="date" v-model="filter.uploadDate" />
      <div class="buttons-group">
        <button @click="applyFilters">筛选</button>
        <button @click="clearFilters">清空筛选</button>
      </div>
    </div>
    <ul>
      <li v-for="image in filteredImages" :key="image.id" @click="selectImage(image)" :class="{ selected: selectedImageId === image.id }">
        {{ image.filename }} ({{ formatDate(image.upload_time) }})
      </li>
    </ul>
    <p v-if="isLoading">加载中...</p>
    <p v-if="!isLoading && filteredImages.length === 0 && !initialLoad">暂无图片，请上传。</p>
    <div class="bottom-actions" style="border-top: 1px solid #eee; padding-top: 15px; margin-top: 15px;">
      <div class="upload-section">
        <input type="file" multiple @change="handleFileUpload" accept="image/png, image/jpeg" ref="fileInput" style="display: none;"/>
        <button @click="triggerFileUpload" :disabled="isUploading" class="full-width-button">{{ isUploading ? '上传中...' : '上传图片' }}</button>
        <p v-if="uploadMessage" :class="{'upload-success': uploadSuccess, 'upload-error': !uploadSuccess}">{{ uploadMessage }}</p>
      </div>
      <div class="action-buttons-group">
        <button @click="deleteSelectedImage" :disabled="!selectedImageId" class="action-button delete-button">删除</button>
        <button @click="exportSelectedImage" :disabled="!selectedImageId" class="action-button export-button">导出</button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, computed, watch } from 'vue';
import axios from 'axios';

const emit = defineEmits(['image-selected', 'image-deleted', 'image-exported']);

const images = ref([]);
const isLoading = ref(false);
const isUploading = ref(false);
const uploadMessage = ref('');
const uploadSuccess = ref(false);
const selectedImageId = ref(null);
const fileInput = ref(null); // 用于触发文件选择
const initialLoad = ref(true); // 标记是否是初始加载

const filter = ref({
  filename: '',
  defectType: '',
  uploadDate: '',
});

// 假设的缺陷类型，实际应用中可能从后端获取或配置
const defectTypes = ref(['气孔', '裂纹', '夹渣', '其他']);

const deleteSelectedImage = async () => {
   if (selectedImageId.value) {
     if (confirm('确定要删除这张图片吗？')) {
       try {
         await axios.delete(`/api/images/${selectedImageId.value}`);
         alert('图片删除成功！');
         fetchImages(); // 刷新图片列表
         emit('image-deleted', selectedImageId.value); // 通知父组件图片已删除
         selectedImageId.value = null; // 删除后清空选中状态
       } catch (error) {
         console.error('删除图片失败:', error);
         alert('删除图片失败，请稍后再试。');
       }
     }
   } else {
     alert('请先选择一张图片进行删除。');
   }
 };
 
 const exportSelectedImage = async () => {
   if (selectedImageId.value) {
     try {
       const response = await axios.get(`/api/images/${selectedImageId.value}/export`, { responseType: 'blob' });
       const contentDisposition = response.headers['content-disposition'];
       let filename = 'exported_image.jpg';
       if (contentDisposition) {
         const filenameMatch = contentDisposition.match(/filename="?(.+)"?/);
         if (filenameMatch && filenameMatch[1]) {
           filename = filenameMatch[1];
         }
       }
       
       const url = window.URL.createObjectURL(new Blob([response.data]));
       const link = document.createElement('a');
       link.href = url;
       link.setAttribute('download', filename);
       document.body.appendChild(link);
       link.click();
       document.body.removeChild(link);
       window.URL.revokeObjectURL(url);
       emit('image-exported', selectedImageId.value); // 通知父组件图片已导出
     } catch (error) {
       console.error('导出图片失败:', error);
       alert('导出图片失败，请稍后再试。');
     }
   }
 };
 
 const fetchImages = async () => {
  isLoading.value = true;
  try {
    const response = await axios.get('/api/images');
    images.value = response.data;
    initialLoad.value = false;
  } catch (error) {
    console.error('获取图片列表失败:', error);
    images.value = []; // 出错时清空列表
  }
  isLoading.value = false;
};

const handleFileUpload = async (event) => {
  const files = event.target.files;
  if (!files.length) return;

  const formData = new FormData();
  for (let i = 0; i < files.length; i++) {
    formData.append('files', files[i]); // 后端需要用 'files' 来接收
  }

  isUploading.value = true;
  uploadMessage.value = '';
  try {
    const response = await axios.post('/api/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    uploadMessage.value = response.data.message || '上传成功！';
    uploadSuccess.value = true;
    fetchImages(); // 上传成功后刷新图片列表
  } catch (error) {
    uploadMessage.value = error.response?.data?.message || '上传失败，请检查文件或联系管理员。';
    uploadSuccess.value = false;
    console.error('上传图片失败:', error);
  }
  isUploading.value = false;
  // 清空文件输入框，以便下次可以选择相同文件
  if (fileInput.value) {
    fileInput.value.value = '';
  }
};

const triggerFileUpload = () => {
  if (fileInput.value) {
    fileInput.value.click();
  }
};

const selectImage = (image) => {
  selectedImageId.value = image.id;
  emit('image-selected', image);
};

const formatDate = (dateString) => {
  if (!dateString) return '';
  const date = new Date(dateString);
  return date.toLocaleDateString('zh-CN') + ' ' + date.toLocaleTimeString('zh-CN');
};

const filteredImages = computed(() => {
  let tempImages = images.value;
  if (filter.value.filename) {
    tempImages = tempImages.filter(img => img.filename.toLowerCase().includes(filter.value.filename.toLowerCase()));
  }
  // 缺陷类型筛选需要后端支持，或在前端获取到所有图片的标注信息后进行筛选
  // 这里暂时只做文件名和日期筛选
  if (filter.value.uploadDate) {
    tempImages = tempImages.filter(img => {
      const imgDate = new Date(img.upload_time).toISOString().split('T')[0];
      return imgDate === filter.value.uploadDate;
    });
  }
  return tempImages;
});

const applyFilters = () => {
  // 计算属性会自动更新，此方法可以用于触发一些额外的逻辑（如果需要）
  console.log('应用筛选:', filter.value);
};

const clearFilters = () => {
  filter.value.filename = '';
  filter.value.defectType = '';
  filter.value.uploadDate = '';
};

onMounted(() => {
  fetchImages();
});

</script>

<style scoped>
.image-list-panel {
  width: 20%; /* 屏幕1/5宽度 */
  height: 100%;
  overflow-y: auto;
  padding: 15px;
  border-right: 1px solid #ddd;
  box-sizing: border-box;
  display: flex;
  flex-direction: column;
}

.upload-section {
  margin-bottom: 15px;
  padding-bottom: 15px;
  border-bottom: 1px solid #eee;
}

.upload-section input[type="file"] {
  /* display: none; */ /* 可以隐藏默认输入框，用按钮触发 */
  margin-bottom: 10px;
  width: 100%;
}

.full-width-button {
  width: 100%;
  padding: 10px;
  margin-top: 10px; /* 按钮之间增加间距 */
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 1em;
}

.upload-section button {
  background-color: #28a745; /* 绿色上传按钮 */
}

.delete-button {
  background-color: #dc3545; /* 红色删除按钮 */
}

.export-button {
  background-color: #007bff; /* 蓝色导出按钮 */
}

.full-width-button:hover {
  opacity: 0.9;
}

.full-width-button:disabled {
  background-color: #ccc;
  cursor: not-allowed;
}

.upload-section button:hover {
  background-color: #218838;
}
.upload-section button:disabled {
  background-color: #ccc;
  cursor: not-allowed;
}
.upload-section button:hover {
  background-color: #218838;
}
.upload-section button:disabled {
  background-color: #ccc;
  cursor: not-allowed;
}

.upload-success {
  color: green;
  font-size: 0.9em;
  margin-top: 5px;
}

.upload-error {
  color: red;
  font-size: 0.9em;
  margin-top: 5px;
}

.filter-section {
   margin-bottom: 15px;
   padding-bottom: 15px;
   border-bottom: 1px solid #eee;
   display: flex;
   flex-wrap: wrap; /* 允许换行 */
   gap: 10px; /* 筛选元素之间的间距 */
 }
 
 .filter-section input,
  .filter-section select {
    flex-grow: 1; /* 允许元素增长 */
    /* width: calc(50% - 5px); */ /* 占据一半宽度减去间距 */
    box-sizing: border-box;
  }
  .filter-section .buttons-group {
    display: flex;
    width: 100%;
    gap: 10px;
  }
  .filter-section .buttons-group button {
    flex-grow: 1;
    width: calc(50% - 5px);
    box-sizing: border-box;
  }
 
 h3 {
   text-align: center; /* 居中标题 */
   margin-bottom: 15px;
 }
 
 .bottom-actions {
  margin-top: auto; /* 将整个底部操作区域推到底部 */
}

.action-buttons-group {
  display: flex;
  gap: 10px; /* 删除和导出按钮之间的间距 */
  margin-top: 10px; /* 与上传按钮的间距 */
}
 
 .action-button {
   flex-grow: 1; /* 按钮占据可用空间 */
   width: calc(50% - 5px); /* 各占一半宽度减去间距 */
   box-sizing: border-box;
   padding: 10px;
   border: none;
   border-radius: 4px;
   cursor: pointer;
   font-size: 1em;
 }
 
 .delete-button {
   background-color: #dc3545; /* 红色删除按钮 */
 }
 
 .export-button {
   background-color: #007bff; /* 蓝色导出按钮 */
 }
 
 .action-button:hover {
   opacity: 0.9;
 }
 
 .action-button:disabled {
   background-color: #ccc;
   cursor: not-allowed;
 }

.filter-section input,
.filter-section select {
  width: calc(100% - 16px); /* 减去padding */
  margin-bottom: 8px;
}

.filter-section button {
  margin-right: 5px;
  padding: 8px 12px;
  background-color: #007bff;
}
.filter-section button:last-child {
  background-color: #6c757d; /* 清空按钮颜色 */
}


ul {
  list-style-type: none;
  padding: 0;
  margin: 0;
  flex-grow: 1; /* 占据剩余空间 */
  overflow-y: auto; /* 列表内部滚动 */
}

li {
  padding: 10px;
  cursor: pointer;
  border-bottom: 1px solid #eee;
  font-size: 0.9em;
}

li:hover {
  background-color: #f0f0f0;
}

li.selected {
  background-color: #007bff;
  color: white;
  font-weight: bold;
}

p {
  text-align: center;
  color: #666;
  font-size: 0.9em;
}
</style>