from typing import List, Dict, Any

class AggregatorAgent:
    def aggregate(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregates the results from multiple chunks.
        """
        if not results:
            return {}

        # For metadata, we can take the result from the first chunk,
        # assuming it's representative of the whole document.
        aggregated_result = {
            "metadata": results[0].get("metadata"),
            "chapters": [],
            "themes": [],
        }

        offset = 0
        for i, result in enumerate(results):
            if i > 0:
                # Calculate the offset based on the previous chunk's text length,
                # minus the overlap.
                # This is a simplified approach. A more robust solution would
                # involve passing chunk information alongside the results.
                offset += len(results[i-1]['text']) - 1000

            if "chapters" in result and result["chapters"]:
                for chapter in result["chapters"].split_points:
                    aggregated_result["chapters"].append(
                        {
                            "position": chapter.position + offset,
                            "title": chapter.title,
                        }
                    )

            if "themes" in result and result["themes"]:
                for theme in result["themes"].split_points:
                    aggregated_result["themes"].append(
                        {
                            "position": theme.position + offset,
                            "theme": theme.theme,
                        }
                    )

        return aggregated_result
