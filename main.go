package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
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
	//DetectToolCallsFromContentWith(content, os.Getenv("MODEL_RUNNER_CHAT_MODEL_QWEN_LATEST"))
	//DetectToolCallsFromContentWith(content, os.Getenv("MODEL_RUNNER_CHAT_MODEL_UNSLOTH_4B"))
	DetectToolCallsFromContentWith(content, os.Getenv("MODEL_RUNNER_CHAT_MODEL_QWEN3_LATEST"))
	//DetectToolCallsFromContentWith(content, os.Getenv("MODEL_RUNNER_CHAT_MODEL_QWEN3_0_6B_F16"))

}

func DetectToolCallsFromContentWith(content string, smallLocalModel string) {
	baseURL := os.Getenv("MODEL_RUNNER_BASE_URL")
	client := openai.NewClient(
		option.WithBaseURL(baseURL),
		option.WithAPIKey(""),
	)

	systemMessageContent, err := GeneratePromptFromToolsCatalog()
	if err != nil {
		log.Fatalf("Error generating system message content: %v", err)
	}

	paramsFirstCompletion := openai.ChatCompletionNewParams{
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.SystemMessage(systemMessageContent),
			openai.UserMessage(content),
		},
		Model:       smallLocalModel,
		Temperature: openai.Opt(0.0),
	}

	completion, err := client.Chat.Completions.New(context.Background(), paramsFirstCompletion)
	if err != nil {
		log.Fatalf("Error creating chat completion: %v", err)
	}
	if len(completion.Choices) == 0 {
		log.Fatal("No choices returned from chat completion")
	}
	result := completion.Choices[0].Message.Content
	if result == "" {
		log.Fatal("No content returned from chat completion")
	}
	fmt.Println("✋ First Result:", result)

	// Make a second completion to force the JSON output format

	paramsSecondCompletion := openai.ChatCompletionNewParams{
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.SystemMessage("Return all function calls wrapped in a container object with a 'function_calls' key."),
			openai.UserMessage(result),
		},
		Model:       smallLocalModel,
		Temperature: openai.Opt(0.0),
		ResponseFormat: openai.ChatCompletionNewParamsResponseFormatUnion{
			OfJSONObject: &openai.ResponseFormatJSONObjectParam{
				Type: "json_object",
			},
		},
	}

	completionNext, err := client.Chat.Completions.New(context.Background(), paramsSecondCompletion)
	if err != nil {
		log.Fatalf("Error creating chat completion for next step: %v", err)
	}
	if len(completionNext.Choices) == 0 {
		log.Fatal("No choices returned from chat completion for next step")
	}
	resultNext := completionNext.Choices[0].Message.Content
	if resultNext == "" {
		log.Fatal("No content returned from chat completion for next step")
	}
	fmt.Println("🚀 Next Result:", resultNext)

	var commands []map[string]any
	errJson := json.Unmarshal([]byte(result), &commands)
	if errJson != nil {
		log.Fatalf("Error unmarshalling JSON result: %v", errJson)
	}
	if len(commands) == 0 {
		log.Fatal("No commands found in the JSON result")
	}
	fmt.Println("Commands found with", smallLocalModel, ":", len(commands))
	for _, command := range commands {
		fmt.Println("  - Command:", command)
	}
}

func GeneratePromptFromToolsCatalog() (string, error) {

	systemContentIntroduction := `You have access to the following tools:`
	// make a JSON String from the content of tools
	toolsJson, err := json.Marshal(GetToolsCatalog())
	if err != nil {
		return "", err
	}
	toolsContent := "[AVAILABLE_TOOLS]" + string(toolsJson) + "[/AVAILABLE_TOOLS]"

	systemContentInstructions := `If the question of the user matched the description of a tool, the tool will be called.
	To call a tool, respond with a JSON object with the following structure: 
	[
		{
			"name": <name of the called tool>,
			"arguments": {
				<name of the argument>: <value of the argument>
			}
		},
	]
	
	search the name of the tool in the list of tools with the Name field
	`
	return systemContentIntroduction + "\n" + toolsContent + "\n" + systemContentInstructions, nil
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
