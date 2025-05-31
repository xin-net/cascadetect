<template>
  <div id="main-layout">
    <ImageListPanel @image-selected="handleImageSelected" />
    <ImageDisplayPanel :image-url="selectedImageUrl" :annotations="currentAnnotations" @update-annotation="handleUpdateAnnotation" @create-annotation="handleCreateAnnotation" @delete-annotation="handleDeleteAnnotation" />
    <AnnotationPanel :annotations="currentAnnotations" @update-annotation="handleUpdateAnnotation" @delete-annotation="handleDeleteAnnotation" @create-annotation="handleCreateAnnotation" :image-id="selectedImageId"/>
  </div>
</template>

<script setup>
import { ref, watch } from 'vue';
import ImageListPanel from './components/ImageListPanel.vue';
import ImageDisplayPanel from './components/ImageDisplayPanel.vue';
import AnnotationPanel from './components/AnnotationPanel.vue';
import axios from 'axios';

const selectedImageUrl = ref('');
const selectedImageId = ref(null);
const currentAnnotations = ref([]);

const fetchAnnotations = async (imageId) => {
  if (!imageId) {
    currentAnnotations.value = [];
    return;
  }
  try {
    // 注意：这里的URL '/api' 会被 vite.config.js 中的代理配置转发到后端的 http://localhost:5000
    const response = await axios.get(`/api/images/${imageId}`);
    selectedImageUrl.value = `/api/uploads/${response.data.filename}`; // 获取图片URL，假设后端在 /uploads 提供静态文件服务
    currentAnnotations.value = response.data.annotations.map(ann => ({ ...ann, isEditing: false }));
  } catch (error) {
    console.error('获取标注失败:', error);
    currentAnnotations.value = [];
    selectedImageUrl.value = '';
  }
};

const handleImageSelected = (image) => {
  selectedImageId.value = image.id;
  // selectedImageUrl.value = `/uploads/${image.filename}`; // 暂时注释，因为后端可能不直接暴露uploads，而是通过API获取
  fetchAnnotations(image.id);
};

const handleUpdateAnnotation = async (annotation) => {
  try {
    await axios.put(`/api/annotations/${annotation.id}`, annotation);
    fetchAnnotations(selectedImageId.value); // 重新获取最新标注
  } catch (error) {
    console.error('更新标注失败:', error);
  }
};

const handleDeleteAnnotation = async (annotationId) => {
  try {
    await axios.delete(`/api/annotations/${annotationId}`);
    fetchAnnotations(selectedImageId.value); // 重新获取最新标注
  } catch (error) {
    console.error('删除标注失败:', error);
  }
};

const handleCreateAnnotation = async (newAnnotation) => {
  try {
    await axios.post(`/api/images/${selectedImageId.value}/annotations`, newAnnotation);
    fetchAnnotations(selectedImageId.value); // 重新获取最新标注
  } catch (error) {
    console.error('创建标注失败:', error);
  }
};

// 监听selectedImageId变化，以便在图片切换时加载新的标注
watch(selectedImageId, (newId) => {
  if (newId) {
    fetchAnnotations(newId);
  } else {
    selectedImageUrl.value = '';
    currentAnnotations.value = [];
  }
});

</script>

<style>
#main-layout {
  display: flex;
  width: 100vw; /* 改为视口宽度 */
  height: 100vh; /* 改为视口高度 */
  max-width: 1920px; /* 限制最大宽度 */
  max-height: 1080px; /* 限制最大高度 */
  margin: auto; /* 居中显示 */
  border: 1px solid #ccc;
  box-sizing: border-box; /* 确保padding和border不增加总宽高 */
  align-items: center; /* 垂直居中对齐子项 */
}

/* 其他全局样式 */
body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
  background-color: #f4f4f4;
}
</style>