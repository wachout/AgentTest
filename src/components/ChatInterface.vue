<template>
  <div class="chat-interface">
    <div class="chat-messages" ref="messageContainer">
      <div v-for="msg in messages" :key="msg.id" :class="['message', msg.sender]">
        <div v-if="msg.type === 'text'" class="message-content">
          {{ msg.content }}
        </div>
        <div v-if="msg.type === 'echart'" class="message-content echart-message-content">
          <EchartRenderer v-for="(chartOption, chartIndex) in msg.chartOptions" :key="chartIndex" :options="chartOption" />
        </div>
        <div v-if="msg.type === 'image'" class="message-content image-message-content">
          <img v-for="(imageBase64, imgIndex) in msg.images" :key="imgIndex" :src="formatBase64Image(imageBase64)" alt="Chat Image" class="chat-image"/>
        </div>
      </div>
    </div>
    <div class="chat-input">
       <input type="text" v-model="userInput" @keyup.enter="handleSendMessage" placeholder="输入您的问题..." :disabled="isLoading" ref="chatInputRef">
      <button @click="handleSendMessage" :disabled="isLoading">
        {{ isLoading ? '发送中...' : '发送' }}
      </button>
    </div>
  </div>
</template>

<script>
import EchartRenderer from './EchartRenderer.vue';
import { getEchartsData, getQueryData } from '@/services/apiService';

export default {
  name: 'ChatInterface',
  components: {
    EchartRenderer
  },
  data() {
    return {
      userInput: '',
      messages: [
        { id: Date.now(), sender: 'bot', type: 'text', content: '您好！有什么可以帮助您的吗？请输入您的问题。' }
      ],
      isLoading: false,
    };
  },
  methods: {
    async handleSendMessage() {
      const query = this.userInput.trim();
      if (query === '') return;

      this.isLoading = true;
      this.addMessage({ sender: 'user', type: 'text', content: query });
      const userQuery = this.userInput; // Store before clearing
      this.userInput = '';
      this.$refs.chatInputRef?.focus(); // Re-focus the input

      const thinkingMsgId = Date.now();
      this.addMessage({ id: thinkingMsgId, sender: 'bot', type: 'text', content: '正在处理，请稍候...' });

      let dataFound = false;

      try {
        // Try to get Echarts data
        const echartsResponse = await getEchartsData(userQuery);
        if (echartsResponse && echartsResponse.echart && echartsResponse.echart.length > 0) {
          this.removeMessageById(thinkingMsgId); // Remove thinking message
          this.addMessage({ sender: 'bot', type: 'echart', chartOptions: echartsResponse.echart });
          dataFound = true;
        }
      } catch (error) {
        console.warn("Error fetching Echarts data:", error.message);
        // Don't show error to user yet, try getQueryData
      }

      try {
        // Try to get Query data (text and/or images)
        // This will run if no echarts were found, or in addition if desired (current logic: if no echarts)
        if (!dataFound) {
          const queryResponse = await getQueryData(userQuery);
          this.removeMessageById(thinkingMsgId); // Remove thinking message if not already removed

          let queryDataProcessed = false;
          if (queryResponse && queryResponse.data) {
            this.addMessage({ sender: 'bot', type: 'text', content: queryResponse.data });
            queryDataProcessed = true;
          }
          if (queryResponse && queryResponse.images && queryResponse.images.length > 0) {
            // If there was also text data, the image should be a new message or combined.
            // For simplicity, let's add images as a separate message if text was also present.
            // Or, if only images, then it's fine.
            this.addMessage({ sender: 'bot', type: 'image', images: queryResponse.images });
            queryDataProcessed = true;
          }

          if (queryDataProcessed) {
            dataFound = true;
          }
        }
      } catch (error) {
        console.error("Error fetching Query data:", error);
        if (!dataFound) { // Only show error if nothing else was found
          this.removeMessageById(thinkingMsgId);
          this.addMessage({ sender: 'bot', type: 'text', content: `处理您的请求时发生错误: ${error.message}` });
        }
      }

      if (!dataFound && this.messages.find(m => m.id === thinkingMsgId)) {
        this.removeMessageById(thinkingMsgId);
        this.addMessage({ sender: 'bot', type: 'text', content: '未能找到相关数据或图表。' });
      }

      this.isLoading = false;
      this.scrollToBottom();
    },

    addMessage(msgDetails) {
      const newMessage = {
        id: msgDetails.id || Date.now() + Math.random(),
        sender: msgDetails.sender,
        type: msgDetails.type,
        content: msgDetails.content || '', // Ensure content exists
        chartOptions: msgDetails.chartOptions || null,
        images: msgDetails.images || null
      };
      this.messages.push(newMessage);
      this.$nextTick(() => {
        this.scrollToBottom();
      });
    },

    removeMessageById(id) {
      this.messages = this.messages.filter(msg => msg.id !== id);
    },

    formatBase64Image(base64String) {
      // The backend returns "b'xxxxxxx'", so we need to remove b' and '
      const cleanedBase64 = base64String.substring(2, base64String.length - 1);
      // Assuming the backend provides the correct mime type or it's a common type like PNG/JPEG
      // For simplicity, let's assume it's PNG. A more robust solution might involve checking the start of the base64 string.
      return `data:image/png;base64,${cleanedBase64}`;
    },

    scrollToBottom() {
      const container = this.$refs.messageContainer;
      if (container) {
        setTimeout(() => {
          container.scrollTop = container.scrollHeight;
        }, 50);
      }
    }
  },
  mounted() {
    this.scrollToBottom();
  }
};
</script>

<style scoped>
/* Add styles for image messages if needed, or adjust existing ones */
.chat-interface {
  display: flex;
  flex-direction: column;
  width: 100%;
  max-width: 800px; /* Adjust as needed */
  height: 70vh; /* Adjust as needed */
  border: 1px solid #ccc;
  border-radius: 8px;
  overflow: hidden;
  margin: 20px auto;
  box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

.chat-messages {
  flex-grow: 1;
  padding: 20px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 10px;
  background-color: #f9f9f9;
}

.message {
  padding: 10px 15px;
  border-radius: 18px;
  max-width: 70%;
  word-wrap: break-word;
  display: flex; /* Use flex for message content alignment */
  flex-direction: column; /* Stack content (like text then image) if needed */
}

.message.user {
  background-color: #007bff;
  color: white;
  align-self: flex-end;
  border-bottom-right-radius: 4px;
}

.message.bot {
  background-color: #e9ecef;
  color: #333;
  align-self: flex-start;
  border-bottom-left-radius: 4px;
}

.message-content {
  white-space: pre-wrap; /* To respect newlines in text */
}

.echart-message-content {
  /* Styles specific to messages containing ECharts */
  padding: 0; /* Remove padding if EchartRenderer has its own */
  background-color: transparent; /* Let EchartRenderer handle background if needed */
}
.image-message-content {
  /* Styles specific to messages containing images */
  padding: 5px;
}

.chat-image {
  max-width: 100%;
  height: auto;
  border-radius: 4px;
  margin-top: 5px; /* Space between multiple images or text and image */
  border: 1px solid #ddd;
}


.chat-input {
  display: flex;
  padding: 10px;
  border-top: 1px solid #ccc;
  background-color: #fff;
}

.chat-input input {
  flex-grow: 1;
  padding: 10px;
  border: 1px solid #ddd;
  border-radius: 20px;
  margin-right: 10px;
  font-size: 1em;
}

.chat-input button {
  padding: 10px 20px;
  background-color: #007bff;
  color: white;
  border: none;
  border-radius: 20px;
  cursor: pointer;
  font-size: 1em;
}

.chat-input button:hover {
  background-color: #0056b3;
}

.chat-input button:disabled {
  background-color: #a0cfff;
  cursor: not-allowed;
}
</style>

<style scoped>
.chat-interface {
  display: flex;
  flex-direction: column;
  width: 100%;
  max-width: 800px; /* Adjust as needed */
  height: 70vh; /* Adjust as needed */
  border: 1px solid #ccc;
  border-radius: 8px;
  overflow: hidden;
  margin: 20px auto;
  box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

.chat-messages {
  flex-grow: 1;
  padding: 20px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 10px;
  background-color: #f9f9f9;
}

.message {
  padding: 10px 15px;
  border-radius: 18px;
  max-width: 70%;
  word-wrap: break-word;
}

.message.user {
  background-color: #007bff;
  color: white;
  align-self: flex-end;
  border-bottom-right-radius: 4px;
}

.message.bot {
  background-color: #e9ecef;
  color: #333;
  align-self: flex-start;
  border-bottom-left-radius: 4px;
}

.message-content {
  white-space: pre-wrap; /* To respect newlines in text */
}

.chat-input {
  display: flex;
  padding: 10px;
  border-top: 1px solid #ccc;
  background-color: #fff;
}

.chat-input input {
  flex-grow: 1;
  padding: 10px;
  border: 1px solid #ddd;
  border-radius: 20px;
  margin-right: 10px;
  font-size: 1em;
}

.chat-input button {
  padding: 10px 20px;
  background-color: #007bff;
  color: white;
  border: none;
  border-radius: 20px;
  cursor: pointer;
  font-size: 1em;
}

.chat-input button:hover {
  background-color: #0056b3;
}
</style>
