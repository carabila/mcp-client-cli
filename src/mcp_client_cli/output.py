import json
from rich.console import Console, ConsoleDimensions
from rich.live import Live
from rich.markdown import Markdown
from rich.prompt import Confirm

class OutputHandler:
    def __init__(self, text_only: bool = False, only_last_message: bool = False):
        self.console = Console()
        self.text_only = text_only
        self.only_last_message = only_last_message
        self.last_message = ""
        if self.text_only:
            self.md = ""
        else:
            self.md = "Thinking...\n"
        self._live = None

    def start(self):
        if not self.text_only:
            self._live = Live(
                Markdown(self.md), 
                vertical_overflow="visible", 
                screen=True,
                console=self.console
            )
            self._live.start()

    def update(self, chunk: any):
        self.md = self._parse_chunk(chunk, self.md)
        if(self.only_last_message and self.text_only):
            # when only_last_message, we print in finish()
            return
        if self.text_only:
            self.console.print(self._parse_chunk(chunk), end="")
        else:
            if self.md.startswith("Thinking...") and not self.md.strip("Thinking...").isspace():
                self.md = self.md.strip("Thinking...").strip()
            partial_md = self._truncate_md_to_fit(self.md, self.console.size)
            self._live.update(Markdown(partial_md), refresh=True)

    def update_error(self, error: Exception):
        import traceback
        error_msg = f"Error: {error}\n\nStack trace:\n```\n{traceback.format_exc()}```"
        self.md += error_msg
        if(self.only_last_message):
            self.console.print(error_msg)
            return
        if self.text_only:
            self.console.print_exception()
        else:
            partial_md = self._truncate_md_to_fit(self.md, self.console.size)
            self._live.update(Markdown(partial_md), refresh=True)

    def stop(self):
        if not self.text_only and self._live:
            self._live.stop()

    def confirm_tool_call(self, config: dict, chunk: any) -> bool:
        if not self._is_tool_call_requested(chunk, config):
            return True

        self.stop()
        is_confirmed = self._ask_tool_call_confirmation()
        if not is_confirmed:
            self.md += "# Tool call denied"
            return False
            
        if not self.text_only:
            self.start()
        return True

    def finish(self):
        self.stop()
        to_print = self.last_message if self.only_last_message else Markdown(self.md)
        if not self.text_only and not self.only_last_message:
            self.console.clear()
            self.console.print(Markdown(self.md))
        if self.only_last_message:
            self.console.print(to_print)

    def _parse_chunk(self, chunk: any, md: str = "") -> str:
        """
        Parse the chunk of agent response.
        It will stream the response as it is received.
        """
        # Handle our new message format
        if isinstance(chunk, dict) and "messages" in chunk:
            messages = chunk["messages"]
            if messages and isinstance(messages, list):
                message = messages[0]
                if isinstance(message, dict):
                    content = message.get("content", "")
                    msg_type = message.get("type", "")
                    
                    if msg_type == "ai":
                        self.last_message += content
                        md += content
                    elif msg_type == "tool":
                        md += f"\n\n**Tool result:** {content}\n"
                    elif msg_type == "error":
                        md += f"\n\n**Error:** {content}\n"
        
        return md

    def _truncate_md_to_fit(self, md: str, dimensions: ConsoleDimensions) -> str:
        """
        Truncate the markdown to fit the console size, with few line safety margin.
        """
        lines = md.splitlines()
        max_lines = dimensions.height - 3  # Safety margin
        fitted_lines = []
        current_height = 0
        code_block_count = 0

        for line in reversed(lines):
            # Calculate wrapped line height, rounding up for safety
            line_height = 1 + len(line) // dimensions.width

            if current_height + line_height > max_lines:
                # If we're breaking in the middle of code blocks, add closing ```
                if code_block_count % 2 == 1:
                    fitted_lines.insert(0, "```")
                break

            fitted_lines.insert(0, line)
            current_height += line_height

            # Track code block markers
            if line.strip() == "```":
                code_block_count += 1

        return '\n'.join(fitted_lines) if fitted_lines else ''

    def _is_tool_call_requested(self, chunk: any, config: dict) -> bool:
        """
        Check if the chunk contains a tool call request and requires confirmation.
        """
        # This would need to be updated based on how we handle tool calls in our new system
        return False

    def _ask_tool_call_confirmation(self) -> bool:
        """
        Ask the user for confirmation to run a tool call.
        """
        self.console.set_alt_screen(True)
        self.console.print(Markdown(self.md))
        self.console.print(f"\n\n")
        is_tool_call_confirmed = Confirm.ask(f"Confirm tool call?", console=self.console)
        self.console.set_alt_screen(False)
        if not is_tool_call_confirmed:
            return False
        return True