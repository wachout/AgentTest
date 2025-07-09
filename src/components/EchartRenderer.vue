<template>
  <div ref="chartContainer" class="echart-container"></div>
</template>

<script>
import * as echarts from 'echarts/core';
import { BarChart, LineChart } from 'echarts/charts';
import { TitleComponent, TooltipComponent, GridComponent, LegendComponent } from 'echarts/components';
import { CanvasRenderer } from 'echarts/renderers';

// Register necessary ECharts components
echarts.use([
  TitleComponent,
  TooltipComponent,
  GridComponent,
  LegendComponent,
  BarChart,
  LineChart,
  CanvasRenderer
]);

export default {
  name: 'EchartRenderer',
  props: {
    options: {
      type: String, // Expecting a stringified ECharts option object
      required: true
    }
  },
  data() {
    return {
      chartInstance: null
    };
  },
  mounted() {
    this.renderChart();
  },
  watch: {
    options() {
      this.renderChart(); // Re-render if options change
    }
  },
  beforeUnmount() {
    if (this.chartInstance) {
      this.chartInstance.dispose();
    }
  },
  methods: {
    renderChart() {
      if (this.chartInstance) {
        this.chartInstance.dispose(); // Dispose previous instance
      }
      if (this.$refs.chartContainer && this.options) {
        try {
          // The backend provides the ECharts option as a string,
          // which might be a JavaScript object literal string.
          // We need to carefully parse it.
          // A safer way than eval is using Function constructor.
          const parsedOptions = new Function(`return ${this.options}`)();

          this.chartInstance = echarts.init(this.$refs.chartContainer);
          this.chartInstance.setOption(parsedOptions);
        } catch (error) {
          console.error("Error parsing or rendering ECharts options:", error, this.options);
          // Optionally, display an error message in the chart container
          this.$refs.chartContainer.innerText = "Error rendering chart. Check console for details.";
        }
      }
    }
  }
};
</script>

<style scoped>
.echart-container {
  width: 100%;
  height: 300px; /* Default height, can be made configurable */
  border: 1px solid #eee;
  border-radius: 4px;
  margin-top: 5px;
}
</style>
