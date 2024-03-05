// Default generic "any" values are for backwards compatibility.
// Replace with "string" when we are comfortable with a breaking change.

import type { BaseCallbackConfig } from "../callbacks/manager.js";
import {
  AIMessage,
  HumanMessage,
  SystemMessage,
  BaseMessage,
  ChatMessage,
  type BaseMessageLike,
  coerceMessageLikeToMessage,
  isBaseMessage,
  MessageContent,
} from "../messages/index.js";
import {
  type ChatPromptValueInterface,
  ChatPromptValue,
} from "../prompt_values.js";
import type { InputValues, PartialValues } from "../utils/types.js";
import { Runnable } from "../runnables/base.js";
import { BaseStringPromptTemplate } from "./string.js";
import {
  BasePromptTemplate,
  type BasePromptTemplateInput,
  type TypedPromptInputValues,
} from "./base.js";
import { PromptTemplate, type ParamsFromFString } from "./prompt.js";
import { ImagePromptTemplate } from "./image.js";
import { parseFString } from "./template.js";

/**
 * Abstract class that serves as a base for creating message prompt
 * templates. It defines how to format messages for different roles in a
 * conversation.
 */
export abstract class BaseMessagePromptTemplate<
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  RunInput extends InputValues = any,
  RunOutput extends BaseMessage[] = BaseMessage[]
> extends Runnable<RunInput, RunOutput> {
  lc_namespace = ["langchain_core", "prompts", "chat"];

  lc_serializable = true;

  abstract inputVariables: Array<Extract<keyof RunInput, string>>;

  /**
   * Method that takes an object of TypedPromptInputValues and returns a
   * promise that resolves to an array of BaseMessage instances.
   * @param values Object of TypedPromptInputValues
   * @returns Formatted array of BaseMessages
   */
  abstract formatMessages(
    values: TypedPromptInputValues<RunInput>
  ): Promise<RunOutput>;

  /**
   * Calls the formatMessages method with the provided input and options.
   * @param input Input for the formatMessages method
   * @param options Optional BaseCallbackConfig
   * @returns Formatted output messages
   */
  async invoke(
    input: RunInput,
    options?: BaseCallbackConfig
  ): Promise<RunOutput> {
    return this._callWithConfig(
      (input: RunInput) => this.formatMessages(input),
      input,
      { ...options, runType: "prompt" }
    );
  }
}

/**
 * Interface for the fields of a MessagePlaceholder.
 */
export interface MessagesPlaceholderFields<T extends string> {
  variableName: T;
  optional?: boolean;
}

/**
 * Class that represents a placeholder for messages in a chat prompt. It
 * extends the BaseMessagePromptTemplate.
 */
export class MessagesPlaceholder<
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    RunInput extends InputValues = any
  >
  extends BaseMessagePromptTemplate<RunInput>
  implements MessagesPlaceholderFields<Extract<keyof RunInput, string>>
{
  static lc_name() {
    return "MessagesPlaceholder";
  }

  variableName: Extract<keyof RunInput, string>;

  optional: boolean;

  constructor(variableName: Extract<keyof RunInput, string>);

  constructor(
    fields: MessagesPlaceholderFields<Extract<keyof RunInput, string>>
  );

  constructor(
    fields:
      | Extract<keyof RunInput, string>
      | MessagesPlaceholderFields<Extract<keyof RunInput, string>>
  ) {
    if (typeof fields === "string") {
      // eslint-disable-next-line no-param-reassign
      fields = { variableName: fields };
    }
    super(fields);
    this.variableName = fields.variableName;
    this.optional = fields.optional ?? false;
  }

  get inputVariables() {
    return [this.variableName];
  }

  validateInputOrThrow(
    input: Array<unknown> | undefined,
    variableName: Extract<keyof RunInput, string>
  ): input is BaseMessage[] {
    if (this.optional && !input) {
      return false;
    } else if (!input) {
      const error = new Error(
        `Error: Field "${variableName}" in prompt uses a MessagesPlaceholder, which expects an array of BaseMessages as an input value. Received: undefined`
      );
      error.name = "InputFormatError";
      throw error;
    }

    let isInputBaseMessage = false;

    if (Array.isArray(input)) {
      isInputBaseMessage = input.every((message) =>
        isBaseMessage(message as BaseMessage)
      );
    } else {
      isInputBaseMessage = isBaseMessage(input as BaseMessage);
    }

    if (!isInputBaseMessage) {
      const readableInput =
        typeof input === "string" ? input : JSON.stringify(input, null, 2);

      const error = new Error(
        `Error: Field "${variableName}" in prompt uses a MessagesPlaceholder, which expects an array of BaseMessages as an input value. Received: ${readableInput}`
      );
      error.name = "InputFormatError";
      throw error;
    }

    return true;
  }

  async formatMessages(
    values: TypedPromptInputValues<RunInput>
  ): Promise<BaseMessage[]> {
    this.validateInputOrThrow(values[this.variableName], this.variableName);

    return values[this.variableName] ?? [];
  }
}

/**
 * Interface for the fields of a MessageStringPromptTemplate.
 */
export interface MessageStringPromptTemplateFields<
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  T extends InputValues = any
> {
  prompt: BaseStringPromptTemplate<T, string>;
}

/**
 * Abstract class that serves as a base for creating message string prompt
 * templates. It extends the BaseMessagePromptTemplate.
 */
export abstract class BaseMessageStringPromptTemplate<
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  RunInput extends InputValues = any
> extends BaseMessagePromptTemplate<RunInput> {
  prompt: BaseStringPromptTemplate<
    InputValues<Extract<keyof RunInput, string>>,
    string
  >;

  constructor(
    prompt: BaseStringPromptTemplate<
      InputValues<Extract<keyof RunInput, string>>
    >
  );

  constructor(
    fields: MessageStringPromptTemplateFields<
      InputValues<Extract<keyof RunInput, string>>
    >
  );

  constructor(
    fields:
      | MessageStringPromptTemplateFields<
          InputValues<Extract<keyof RunInput, string>>
        >
      | BaseStringPromptTemplate<
          InputValues<Extract<keyof RunInput, string>>,
          string
        >
  ) {
    if (!("prompt" in fields)) {
      // eslint-disable-next-line no-param-reassign
      fields = { prompt: fields };
    }
    super(fields);
    this.prompt = fields.prompt;
  }

  get inputVariables() {
    return this.prompt.inputVariables;
  }

  abstract format(
    values: TypedPromptInputValues<RunInput>
  ): Promise<BaseMessage>;

  async formatMessages(
    values: TypedPromptInputValues<RunInput>
  ): Promise<BaseMessage[]> {
    return [await this.format(values)];
  }
}

/**
 * Abstract class that serves as a base for creating chat prompt
 * templates. It extends the BasePromptTemplate.
 */
export abstract class BaseChatPromptTemplate<
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  RunInput extends InputValues = any,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  PartialVariableName extends string = any
> extends BasePromptTemplate<
  RunInput,
  ChatPromptValueInterface,
  PartialVariableName
> {
  constructor(input: BasePromptTemplateInput<RunInput, PartialVariableName>) {
    super(input);
  }

  abstract formatMessages(
    values: TypedPromptInputValues<RunInput>
  ): Promise<BaseMessage[]>;

  async format(values: TypedPromptInputValues<RunInput>): Promise<string> {
    return (await this.formatPromptValue(values)).toString();
  }

  async formatPromptValue(
    values: TypedPromptInputValues<RunInput>
  ): Promise<ChatPromptValueInterface> {
    const resultMessages = await this.formatMessages(values);
    return new ChatPromptValue(resultMessages);
  }
}

/**
 * Interface for the fields of a ChatMessagePromptTemplate.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export interface ChatMessagePromptTemplateFields<T extends InputValues = any>
  extends MessageStringPromptTemplateFields<T> {
  role: string;
}

/**
 * Class that represents a chat message prompt template. It extends the
 * BaseMessageStringPromptTemplate.
 */
export class ChatMessagePromptTemplate<
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  RunInput extends InputValues = any
> extends BaseMessageStringPromptTemplate<RunInput> {
  static lc_name() {
    return "ChatMessagePromptTemplate";
  }

  role: string;

  constructor(
    prompt: BaseStringPromptTemplate<
      InputValues<Extract<keyof RunInput, string>>
    >,
    role: string
  );

  constructor(
    fields: ChatMessagePromptTemplateFields<
      InputValues<Extract<keyof RunInput, string>>
    >
  );

  constructor(
    fields:
      | ChatMessagePromptTemplateFields<
          InputValues<Extract<keyof RunInput, string>>
        >
      | BaseStringPromptTemplate<InputValues<Extract<keyof RunInput, string>>>,
    role?: string
  ) {
    if (!("prompt" in fields)) {
      // eslint-disable-next-line no-param-reassign, @typescript-eslint/no-non-null-assertion
      fields = { prompt: fields, role: role! };
    }
    super(fields);
    this.role = fields.role;
  }

  async format(values: RunInput): Promise<BaseMessage> {
    return new ChatMessage(await this.prompt.format(values), this.role);
  }

  static fromTemplate(template: string, role: string) {
    return new this(PromptTemplate.fromTemplate(template), role);
  }
}

interface _TextTemplateParam {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  text?: string | Record<string, any>;
}

interface _ImageTemplateParam {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  image_url?: string | Record<string, any>;
}

type MessageClass =
  | typeof HumanMessage
  | typeof AIMessage
  | typeof SystemMessage;

type ChatMessageClass = typeof ChatMessage;

class _StringImageMessagePromptTemplate<
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  RunInput extends InputValues = any,
  RunOutput extends BaseMessage[] = BaseMessage[]
> extends BaseMessagePromptTemplate<RunInput, RunOutput> {
  lc_namespace = ["langchain_core", "prompts", "chat"];

  lc_serializable = true;

  inputVariables: Array<Extract<keyof RunInput, string>> = [];

  additionalOptions: Record<string, unknown> = {};

  prompt:
    | BaseStringPromptTemplate<
        InputValues<Extract<keyof RunInput, string>>,
        string
      >
    | Array<
        | BaseStringPromptTemplate<
            InputValues<Extract<keyof RunInput, string>>,
            string
          >
        | ImagePromptTemplate<
            InputValues<Extract<keyof RunInput, string>>,
            string
          >
        | MessageStringPromptTemplateFields<
            InputValues<Extract<keyof RunInput, string>>
          >
      >;

  protected messageClass?: MessageClass;

  static _messageClass(): MessageClass {
    throw new Error(
      "Can not invoke _messageClass from inside _StringImageMessagePromptTemplate"
    );
  }

  // ChatMessage contains role field, others don't.
  // Because of this, we have a separate class property for ChatMessage.
  protected chatMessageClass?: ChatMessageClass;

  constructor(
    /** @TODO When we come up with a better way to type prompt templates, fix this */
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    fields: any,
    additionalOptions?: Record<string, unknown>
  ) {
    if (!("prompt" in fields)) {
      // eslint-disable-next-line no-param-reassign
      fields = { prompt: fields };
    }
    super(fields);
    this.prompt = fields.prompt;
    if (Array.isArray(this.prompt)) {
      let inputVariables: Extract<keyof RunInput, string>[] = [];
      this.prompt.forEach((prompt) => {
        if ("inputVariables" in prompt) {
          inputVariables = inputVariables.concat(prompt.inputVariables);
        }
      });
      this.inputVariables = inputVariables;
    } else {
      this.inputVariables = this.prompt.inputVariables;
    }
    this.additionalOptions = additionalOptions ?? this.additionalOptions;
  }

  createMessage(content: MessageContent) {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const constructor = this.constructor as any;
    if (constructor._messageClass()) {
      const MsgClass = constructor._messageClass();
      return new MsgClass({ content });
    } else if (constructor.chatMessageClass) {
      const MsgClass = constructor.chatMessageClass();
      // Assuming ChatMessage constructor also takes a content argument
      return new MsgClass({
        content,
        role: this.getRoleFromMessageClass(MsgClass.lc_name()),
      });
    } else {
      throw new Error("No message class defined");
    }
  }

  getRoleFromMessageClass(name: string) {
    switch (name) {
      case "HumanMessage":
        return "human";
      case "AIMessage":
        return "ai";
      case "SystemMessage":
        return "system";
      case "ChatMessage":
        return "chat";
      default:
        throw new Error("Invalid message class name");
    }
  }

  static fromTemplate(
    template: string | Array<string | _TextTemplateParam | _ImageTemplateParam>,
    additionalOptions?: Record<string, unknown>
  ) {
    if (typeof template === "string") {
      return new this(PromptTemplate.fromTemplate(template));
    }
    const prompt: Array<
      PromptTemplate<InputValues> | ImagePromptTemplate<InputValues>
    > = [];
    for (const item of template) {
      if (
        typeof item === "string" ||
        (typeof item === "object" && "text" in item)
      ) {
        let text = "";
        if (typeof item === "string") {
          text = item;
        } else if (typeof item.text === "string") {
          text = item.text ?? "";
        }
        prompt.push(PromptTemplate.fromTemplate(text));
      } else if (typeof item === "object" && "image_url" in item) {
        let imgTemplate = item.image_url ?? "";
        let imgTemplateObject: ImagePromptTemplate<InputValues>;
        let inputVariables: string[] = [];
        if (typeof imgTemplate === "string") {
          const parsedTemplate = parseFString(imgTemplate);
          const variables = parsedTemplate.flatMap((item) =>
            item.type === "variable" ? [item.name] : []
          );

          if ((variables?.length ?? 0) > 0) {
            if (variables.length > 1) {
              throw new Error(
                `Only one format variable allowed per image template.\nGot: ${variables}\nFrom: ${imgTemplate}`
              );
            }
            inputVariables = [variables[0]];
          } else {
            inputVariables = [];
          }

          imgTemplate = { url: imgTemplate };
          imgTemplateObject = new ImagePromptTemplate<InputValues>({
            template: imgTemplate,
            inputVariables,
          });
        } else if (typeof imgTemplate === "object") {
          if ("url" in imgTemplate) {
            const parsedTemplate = parseFString(imgTemplate.url);
            inputVariables = parsedTemplate.flatMap((item) =>
              item.type === "variable" ? [item.name] : []
            );
          } else {
            inputVariables = [];
          }
          imgTemplateObject = new ImagePromptTemplate<InputValues>({
            template: imgTemplate,
            inputVariables,
          });
        } else {
          throw new Error("Invalid image template");
        }
        prompt.push(imgTemplateObject);
      }
    }
    return new this({ prompt, additionalOptions });
  }

  async format(input: TypedPromptInputValues<RunInput>): Promise<BaseMessage> {
    // eslint-disable-next-line no-instanceof/no-instanceof
    if (this.prompt instanceof BaseStringPromptTemplate) {
      const text = await this.prompt.format(input);

      return this.createMessage(text);
    } else {
      const content: MessageContent = [];
      for (const prompt of this.prompt) {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        let inputs: Record<string, any> = {};
        if (!("inputVariables" in prompt)) {
          throw new Error(
            `Prompt ${prompt} does not have inputVariables defined.`
          );
        }
        for (const item of prompt.inputVariables) {
          if (!inputs) {
            inputs = { [item]: input[item] };
          }
          inputs = { ...inputs, [item]: input[item] };
        }
        // eslint-disable-next-line no-instanceof/no-instanceof
        if (prompt instanceof BaseStringPromptTemplate) {
          const formatted = await prompt.format(
            inputs as TypedPromptInputValues<RunInput>
          );
          content.push({ type: "text", text: formatted });
          /** @TODO replace this */
          // eslint-disable-next-line no-instanceof/no-instanceof
        } else if (prompt instanceof ImagePromptTemplate) {
          const formatted = await prompt.format(
            inputs as TypedPromptInputValues<RunInput>
          );
          content.push({ type: "image_url", image_url: formatted });
        }
      }

      return this.createMessage(content);
    }
  }

  async formatMessages(values: RunInput): Promise<RunOutput> {
    return [await this.format(values)] as BaseMessage[] as RunOutput;
  }
}

/**
 * Class that represents a human message prompt template. It extends the
 * BaseMessageStringPromptTemplate.
 * @example
 * ```typescript
 * const message = HumanMessagePromptTemplate.fromTemplate("{text}");
 * const formatted = await message.format({ text: "Hello world!" });
 *
 * const chatPrompt = ChatPromptTemplate.fromMessages([message]);
 * const formattedChatPrompt = await chatPrompt.invoke({
 *   text: "Hello world!",
 * });
 * ```
 */
export class HumanMessagePromptTemplate<
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  RunInput extends InputValues = any
> extends _StringImageMessagePromptTemplate<RunInput> {
  static _messageClass(): typeof HumanMessage {
    return HumanMessage;
  }

  static lc_name() {
    return "HumanMessagePromptTemplate";
  }
}

/**
 * Class that represents an AI message prompt template. It extends the
 * BaseMessageStringPromptTemplate.
 */
export class AIMessagePromptTemplate<
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  RunInput extends InputValues = any
> extends _StringImageMessagePromptTemplate<RunInput> {
  static _messageClass(): typeof AIMessage {
    return AIMessage;
  }

  static lc_name() {
    return "AIMessagePromptTemplate";
  }
}

/**
 * Class that represents a system message prompt template. It extends the
 * BaseMessageStringPromptTemplate.
 * @example
 * ```typescript
 * const message = SystemMessagePromptTemplate.fromTemplate("{text}");
 * const formatted = await message.format({ text: "Hello world!" });
 *
 * const chatPrompt = ChatPromptTemplate.fromMessages([message]);
 * const formattedChatPrompt = await chatPrompt.invoke({
 *   text: "Hello world!",
 * });
 * ```
 */
export class SystemMessagePromptTemplate<
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  RunInput extends InputValues = any
> extends _StringImageMessagePromptTemplate<RunInput> {
  static _messageClass(): typeof SystemMessage {
    return SystemMessage;
  }

  static lc_name() {
    return "SystemMessagePromptTemplate";
  }
}

/**
 * Interface for the input of a ChatPromptTemplate.
 */
export interface ChatPromptTemplateInput<
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  RunInput extends InputValues = any,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  PartialVariableName extends string = any
> extends BasePromptTemplateInput<RunInput, PartialVariableName> {
  /**
   * The prompt messages
   */
  promptMessages: Array<BaseMessagePromptTemplate | BaseMessage>;

  /**
   * Whether to try validating the template on initialization
   *
   * @defaultValue `true`
   */
  validateTemplate?: boolean;
}

export type BaseMessagePromptTemplateLike =
  | BaseMessagePromptTemplate
  | BaseMessageLike;

function _isBaseMessagePromptTemplate(
  baseMessagePromptTemplateLike: BaseMessagePromptTemplateLike
): baseMessagePromptTemplateLike is BaseMessagePromptTemplate {
  return (
    typeof (baseMessagePromptTemplateLike as BaseMessagePromptTemplate)
      .formatMessages === "function"
  );
}

function _coerceMessagePromptTemplateLike(
  messagePromptTemplateLike: BaseMessagePromptTemplateLike
): BaseMessagePromptTemplate | BaseMessage {
  if (
    _isBaseMessagePromptTemplate(messagePromptTemplateLike) ||
    isBaseMessage(messagePromptTemplateLike)
  ) {
    return messagePromptTemplateLike;
  }
  const message = coerceMessageLikeToMessage(messagePromptTemplateLike);
  if (message._getType() === "human") {
    return HumanMessagePromptTemplate.fromTemplate(message.content);
  } else if (message._getType() === "ai") {
    return AIMessagePromptTemplate.fromTemplate(message.content);
  } else if (message._getType() === "system") {
    return SystemMessagePromptTemplate.fromTemplate(message.content);
  } else if (ChatMessage.isInstance(message)) {
    return ChatMessagePromptTemplate.fromTemplate(
      message.content as string,
      message.role
    );
  } else {
    throw new Error(
      `Could not coerce message prompt template from input. Received message type: "${message._getType()}".`
    );
  }
}

function isMessagesPlaceholder(
  x: BaseMessagePromptTemplate | BaseMessage
): x is MessagesPlaceholder {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  return (x.constructor as any).lc_name() === "MessagesPlaceholder";
}

/**
 * Class that represents a chat prompt. It extends the
 * BaseChatPromptTemplate and uses an array of BaseMessagePromptTemplate
 * instances to format a series of messages for a conversation.
 * @example
 * ```typescript
 * const message = SystemMessagePromptTemplate.fromTemplate("{text}");
 * const chatPrompt = ChatPromptTemplate.fromMessages([
 *   ["ai", "You are a helpful assistant."],
 *   message,
 * ]);
 * const formattedChatPrompt = await chatPrompt.invoke({
 *   text: "Hello world!",
 * });
 * ```
 */
export class ChatPromptTemplate<
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    RunInput extends InputValues = any,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    PartialVariableName extends string = any
  >
  extends BaseChatPromptTemplate<RunInput, PartialVariableName>
  implements ChatPromptTemplateInput<RunInput, PartialVariableName>
{
  static lc_name() {
    return "ChatPromptTemplate";
  }

  get lc_aliases() {
    return {
      promptMessages: "messages",
    };
  }

  promptMessages: Array<BaseMessagePromptTemplate | BaseMessage>;

  validateTemplate = true;

  constructor(input: ChatPromptTemplateInput<RunInput, PartialVariableName>) {
    super(input);
    Object.assign(this, input);

    if (this.validateTemplate) {
      const inputVariablesMessages = new Set<string>();
      for (const promptMessage of this.promptMessages) {
        // eslint-disable-next-line no-instanceof/no-instanceof
        if (promptMessage instanceof BaseMessage) continue;
        for (const inputVariable of promptMessage.inputVariables) {
          inputVariablesMessages.add(inputVariable);
        }
      }

      const totalInputVariables = this.inputVariables as string[];
      const inputVariablesInstance = new Set(
        this.partialVariables
          ? totalInputVariables.concat(Object.keys(this.partialVariables))
          : totalInputVariables
      );
      const difference = new Set(
        [...inputVariablesInstance].filter(
          (x) => !inputVariablesMessages.has(x)
        )
      );
      if (difference.size > 0) {
        throw new Error(
          `Input variables \`${[
            ...difference,
          ]}\` are not used in any of the prompt messages.`
        );
      }
      const otherDifference = new Set(
        [...inputVariablesMessages].filter(
          (x) => !inputVariablesInstance.has(x)
        )
      );
      if (otherDifference.size > 0) {
        throw new Error(
          `Input variables \`${[
            ...otherDifference,
          ]}\` are used in prompt messages but not in the prompt template.`
        );
      }
    }
  }

  _getPromptType(): "chat" {
    return "chat";
  }

  private async _parseImagePrompts(
    message: BaseMessage,
    inputValues: InputValues<
      PartialVariableName | Extract<keyof RunInput, string>
    >
  ): Promise<BaseMessage> {
    if (typeof message.content === "string") {
      return message;
    }
    const formattedMessageContent = await Promise.all(
      message.content.map(async (item) => {
        if (item.type !== "image_url") {
          return item;
        }

        let imageUrl = "";
        if (typeof item.image_url === "string") {
          imageUrl = item.image_url;
        } else {
          imageUrl = item.image_url.url;
        }

        const promptTemplatePlaceholder = PromptTemplate.fromTemplate(imageUrl);
        const formattedUrl = await promptTemplatePlaceholder.format(
          inputValues
        );

        if (typeof item.image_url !== "string" && "url" in item.image_url) {
          // eslint-disable-next-line no-param-reassign
          item.image_url.url = formattedUrl;
        } else {
          // eslint-disable-next-line no-param-reassign
          item.image_url = formattedUrl;
        }
        return item;
      })
    );
    // eslint-disable-next-line no-param-reassign
    message.content = formattedMessageContent;
    return message;
  }

  async formatMessages(
    values: TypedPromptInputValues<RunInput>
  ): Promise<BaseMessage[]> {
    const allValues = await this.mergePartialAndUserVariables(values);
    let resultMessages: BaseMessage[] = [];

    for (const promptMessage of this.promptMessages) {
      // eslint-disable-next-line no-instanceof/no-instanceof
      if (promptMessage instanceof BaseMessage) {
        resultMessages.push(
          await this._parseImagePrompts(promptMessage, allValues)
        );
      } else {
        const inputValues = promptMessage.inputVariables.reduce(
          (acc, inputVariable) => {
            if (
              !(inputVariable in allValues) &&
              !(isMessagesPlaceholder(promptMessage) && promptMessage.optional)
            ) {
              throw new Error(
                `Missing value for input variable \`${inputVariable.toString()}\``
              );
            }
            acc[inputVariable] = allValues[inputVariable];
            return acc;
          },
          {} as InputValues
        );
        const message = await promptMessage.formatMessages(inputValues);
        resultMessages = resultMessages.concat(message);
      }
    }
    return resultMessages;
  }

  async partial<NewPartialVariableName extends string>(
    values: PartialValues<NewPartialVariableName>
  ) {
    // This is implemented in a way it doesn't require making
    // BaseMessagePromptTemplate aware of .partial()
    const newInputVariables = this.inputVariables.filter(
      (iv) => !(iv in values)
    ) as Exclude<Extract<keyof RunInput, string>, NewPartialVariableName>[];
    const newPartialVariables = {
      ...(this.partialVariables ?? {}),
      ...values,
    } as PartialValues<PartialVariableName | NewPartialVariableName>;
    const promptDict = {
      ...this,
      inputVariables: newInputVariables,
      partialVariables: newPartialVariables,
    };
    return new ChatPromptTemplate<
      InputValues<
        Exclude<Extract<keyof RunInput, string>, NewPartialVariableName>
      >
    >(promptDict);
  }

  /**
   * Load prompt template from a template f-string
   */
  static fromTemplate<
    // eslint-disable-next-line @typescript-eslint/ban-types
    RunInput extends InputValues = Symbol,
    T extends string = string
  >(template: T) {
    const prompt = PromptTemplate.fromTemplate(template);
    const humanTemplate = new HumanMessagePromptTemplate({ prompt });
    return this.fromMessages<
      // eslint-disable-next-line @typescript-eslint/ban-types
      RunInput extends Symbol ? ParamsFromFString<T> : RunInput
    >([humanTemplate]);
  }

  /**
   * Create a chat model-specific prompt from individual chat messages
   * or message-like tuples.
   * @param promptMessages Messages to be passed to the chat model
   * @returns A new ChatPromptTemplate
   */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  static fromMessages<RunInput extends InputValues = any>(
    promptMessages: (
      | ChatPromptTemplate<InputValues, string>
      | BaseMessagePromptTemplateLike
    )[]
  ): ChatPromptTemplate<RunInput> {
    const flattenedMessages = promptMessages.reduce(
      (acc: Array<BaseMessagePromptTemplate | BaseMessage>, promptMessage) =>
        acc.concat(
          // eslint-disable-next-line no-instanceof/no-instanceof
          promptMessage instanceof ChatPromptTemplate
            ? promptMessage.promptMessages
            : [_coerceMessagePromptTemplateLike(promptMessage)]
        ),
      []
    );
    const flattenedPartialVariables = promptMessages.reduce(
      (acc, promptMessage) =>
        // eslint-disable-next-line no-instanceof/no-instanceof
        promptMessage instanceof ChatPromptTemplate
          ? Object.assign(acc, promptMessage.partialVariables)
          : acc,
      Object.create(null) as PartialValues
    );
    const inputVariables = new Set<string>();
    for (const promptMessage of flattenedMessages) {
      // eslint-disable-next-line no-instanceof/no-instanceof
      if (promptMessage instanceof BaseMessage) continue;
      for (const inputVariable of promptMessage.inputVariables) {
        if (inputVariable in flattenedPartialVariables) {
          continue;
        }
        inputVariables.add(inputVariable);
      }
    }
    return new ChatPromptTemplate<RunInput>({
      inputVariables: [...inputVariables] as Extract<keyof RunInput, string>[],
      promptMessages: flattenedMessages,
      partialVariables: flattenedPartialVariables,
    });
  }

  /** @deprecated Renamed to .fromMessages */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  static fromPromptMessages<RunInput extends InputValues = any>(
    promptMessages: (
      | ChatPromptTemplate<InputValues, string>
      | BaseMessagePromptTemplateLike
    )[]
  ): ChatPromptTemplate<RunInput> {
    return this.fromMessages(promptMessages);
  }
}
