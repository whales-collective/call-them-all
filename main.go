package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/shared"
)

func main() {
 
	content := `
		Make a Vulcan salute to Spock
		Say Hello to John Doe
		Why the sky is blue?
		Add 10 and 32
		Make a Vulcan salute to Bob Morane
		Say Hello to Jane Doe
		Who is Jean-Luc Picard?
		I'm Philippe
		Add 5 and 37
		Make a Vulcan salute to Sam
		Say hello to Alice and then make a vulcan salut to Bob
	`

	//🟢 DetectToolCallsFromContentWith(content, os.Getenv("MODEL_RUNNER_CHAT_MODEL_QWEN_LATEST"))
	DetectToolCallsFromContentWith(content, os.Getenv("MODEL_RUNNER_CHAT_MODEL_QWEN2_5_1_5B_F16"))
	//🟢 DetectToolCallsFromContentWith(content, os.Getenv("MODEL_RUNNER_CHAT_MODEL_QWEN2_5_3B_F16"))
	//DetectToolCallsFromContentWith(content, os.Getenv("MODEL_RUNNER_CHAT_MODEL_GEMMA3_LATEST"))
	//DetectToolCallsFromContentWith(content, os.Getenv("MODEL_RUNNER_CHAT_MODEL_QWEN3_0_6B_Q4_K_M"))

}

func DetectToolCallsFromContentWith(content string, smallLocalModel string) {
	baseURL := os.Getenv("MODEL_RUNNER_BASE_URL")
	client := openai.NewClient(
		option.WithBaseURL(baseURL),
		option.WithAPIKey(""),
	)

	//systemMessageContent, err := GeneratePromptFromToolsCatalog()
	systemMessageContent, err := GeneratePromptFromToolsCatalog()

	if err != nil {
		log.Fatalf("Error generating system message content: %v", err)
	}

	responseFormat := openai.ChatCompletionNewParamsResponseFormatUnion{
		OfJSONSchema: &openai.ResponseFormatJSONSchemaParam{
			Type: "json_schema",
			JSONSchema: shared.ResponseFormatJSONSchemaJSONSchemaParam{
				Name:        "function_calls",
				Description: openai.String("Function calls data structure"),
				Schema: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"function_calls": map[string]any{
							"type": "array",
							"items": map[string]any{
								"type": "object",
								"properties": map[string]any{
									"name": map[string]any{
										"type":        "string",
										"description": "The name of the function to call",
									},
									"arguments": map[string]any{
										"type":        "object",
										"description": "The arguments to pass to the function",
									},
								},
								"required":             []string{"name", "arguments"},
								"additionalProperties": false,
							},
							"description": "Array of function calls to execute",
						},
					},
					"required":             []string{"function_calls"},
					"additionalProperties": false,
				},
			},
		},
	}

	paramsCompletion := openai.ChatCompletionNewParams{
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.SystemMessage(systemMessageContent),
			openai.UserMessage(content),
		},
		Model:          smallLocalModel,
		Temperature:    openai.Opt(0.0),
		ResponseFormat: responseFormat,
	}

	completion, err := client.Chat.Completions.New(context.Background(), paramsCompletion)
	if err != nil {
		log.Fatalf("Error creating chat completion: %v", err)
	}
	if len(completion.Choices) == 0 {
		log.Fatal("No choices returned from chat completion")
	}
	result := completion.Choices[0].Message.Content
	if result == "" {
		log.Fatal("No tool calls detected")
	}
	fmt.Println("✋ JSON String Result:", result)

	type Command struct {
		Name      string         `json:"name"`
		Arguments map[string]any `json:"arguments"`
	}

	type FunctionCalls struct {
		FunctionCalls []Command `json:"function_calls"`
	}

	var commands FunctionCalls

	errJson := json.Unmarshal([]byte(result), &commands)
	if errJson != nil {
		log.Fatalf("Error unmarshalling JSON result: %v", errJson)
	}
	if len(commands.FunctionCalls) == 0 {
		log.Fatal("No commands found in the JSON result")
	}
	fmt.Println("Commands found with", smallLocalModel, ":", len(commands.FunctionCalls))
	for _, command := range commands.FunctionCalls {
		fmt.Println("  - Command:", command)
	}
}

func GetToolsCatalog() []openai.ChatCompletionToolParam {

	vulcanSaluteTool := openai.ChatCompletionToolParam{
		Function: openai.FunctionDefinitionParam{
			Name:        "vulcan_salute",
			Description: openai.String("Give a vulcan salute to the given person name"),
			Parameters: openai.FunctionParameters{
				"type": "object",
				"properties": map[string]interface{}{
					"name": map[string]string{
						"type": "string",
					},
				},
				"required": []string{"name"},
			},
		},
	}

	sayHelloTool := openai.ChatCompletionToolParam{
		Function: openai.FunctionDefinitionParam{
			Name:        "say_hello",
			Description: openai.String("Say hello to the given person name"),
			Parameters: openai.FunctionParameters{
				"type": "object",
				"properties": map[string]interface{}{
					"name": map[string]string{
						"type": "string",
					},
				},
				"required": []string{"name"},
			},
		},
	}

	additionTool := openai.ChatCompletionToolParam{
		Function: openai.FunctionDefinitionParam{
			Name:        "addition",
			Description: openai.String("Add two numbers together"),
			Parameters: openai.FunctionParameters{
				"type": "object",
				"properties": map[string]interface{}{
					"number1": map[string]string{
						"type": "number",
					},
					"number2": map[string]string{
						"type": "number",
					},
				},
				"required": []string{"number1", "number2"},
			},
		},
	}

	tools := []openai.ChatCompletionToolParam{
		vulcanSaluteTool, sayHelloTool, additionTool,
	}
	return tools
}

func GeneratePromptFromToolsCatalog() (string, error) {
	systemContentIntroduction := `You are an AI assistant with access to various tools. Your task is to analyze user input and identify ALL possible tool calls that can be made.
	IMPORTANT: You must process the ENTIRE user input and identify ALL tool calls, not just the first few. Each line or request in the user input should be analyzed separately.
	You have access to the following tools:
	`

	// make a JSON String from the content of tools
	toolsJson, err := json.Marshal(GetToolsCatalog())
	if err != nil {
		return "", err
	}
	toolsContent := "\n[AVAILABLE_TOOLS]\n" + string(toolsJson) + "\n[/AVAILABLE_TOOLS]\n"

	systemContentInstructions := `INSTRUCTIONS:
	1. Read the ENTIRE user input carefully
	2. Process each line/request separately
	3. For each request, check if it matches any tool description
	4. If multiple tool calls are needed, include ALL of them in your response
	5. NEVER stop processing until you've analyzed the complete input

	TOOL MATCHING RULES:
	- Match tool calls based on the "description" field of each tool
	- Use the exact "name" field from the tool definition
	- Provide all required arguments as specified in the tool's parameters

	RESPONSE FORMAT:
	When you find tool calls, respond with a JSON array containing ALL identified tool calls:
	[
		{
			"name": "<exact_tool_name_from_catalog>",
			"arguments": {
				"<parameter_name>": "<parameter_value>"
			}
		},
		{
			"name": "<next_tool_name>",
			"arguments": {
				"<parameter_name>": "<parameter_value>"
			}
		}
	]

	EXAMPLES:
	Input: "Say hello to John. Add 5 and 10. Make vulcan salute to Spock."
	Output: [
		{"name": "send_message", "arguments": {"name": "John"}},
		{"name": "operation", "arguments": {"number1": 5, "number2": 10, "number3": 8}},
		{"name": "greetings", "arguments": {"name": "Jane"}}
	]

	If no tool calls are found, respond with an empty array: []

	CRITICAL: You must analyze the COMPLETE user input and identify ALL possible tool calls. Do not stop after finding the first few matches.
	`

	return systemContentIntroduction + toolsContent + systemContentInstructions, nil
}
