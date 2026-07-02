import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union


@dataclass
class Token:
    type: str
    value: str
    position: int


@dataclass
class Program:
    declarations: List["Declaration"]
    statement: "IfElse"


@dataclass
class Declaration:
    data_type: str
    name: str
    value: Union[int, bool, str]


@dataclass
class IfElse:
    condition: "BinaryExpression"
    then_branch: "Assignment"
    else_branch: "Assignment"


@dataclass
class Assignment:
    name: str
    expression: Union["BinaryExpression", int, bool, str]


@dataclass
class BinaryExpression:
    left: Union[str, int, bool, "BinaryExpression"]
    operator: str
    right: Union[str, int, bool, "BinaryExpression"]


class Lexer:
    TOKEN_SPECIFICATION: List[Tuple[str, str]] = [
        ("INT_TYPE", r"\bint\b"),
        ("BOOL_TYPE", r"\bbool\b"),
        ("IF", r"\bif\b"),
        ("ELSE", r"\belse\b"),
        ("TRUE", r"\btrue\b"),
        ("FALSE", r"\bfalse\b"),
        ("NUMBER", r"\d+"),
        ("OPERATOR", r"==|!=|<=|>=|[+\-*/<>]"),
        ("ASSIGN", r"="),
        ("SEMICOLON", r";"),
        ("LPAREN", r"\("),
        ("RPAREN", r"\)"),
        ("LBRACE", r"\{"),
        ("RBRACE", r"\}"),
        ("IDENTIFIER", r"[A-Za-z_][A-Za-z0-9_]*"),
        ("SKIP", r"[ \t\r\n]+"),
        ("MISMATCH", r"."),
    ]

    def __init__(self, source_code: str):
        self.source_code = source_code
        self.regex = re.compile(
            "|".join(f"(?P<{token_type}>{pattern})" for token_type, pattern in self.TOKEN_SPECIFICATION)
        )

    def tokenize(self) -> List[Token]:
        tokens: List[Token] = []

        for match in self.regex.finditer(self.source_code):
            token_type = match.lastgroup
            value = match.group()
            position = match.start()

            if token_type == "SKIP":
                continue
            if token_type == "MISMATCH":
                raise SyntaxError(f"Karakter tidak dikenal '{value}' pada posisi {position}.")

            tokens.append(Token(token_type, value, position))

        tokens.append(Token("EOF", "", len(self.source_code)))
        return tokens


class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.current = 0

    def parse(self) -> Program:
        declarations: List[Declaration] = []

        while self.peek().type in ("INT_TYPE", "BOOL_TYPE"):
            declarations.append(self.parse_declaration())

        statement = self.parse_if_else()
        self.consume("EOF", "Program memiliki token tambahan setelah blok if-else.")
        return Program(declarations, statement)

    def parse_declaration(self) -> Declaration:
        type_token = self.advance()
        name = self.consume("IDENTIFIER", "Deklarasi membutuhkan nama variabel.").value
        self.consume("ASSIGN", "Deklarasi membutuhkan operator '='.")
        value = self.parse_literal()
        self.consume("SEMICOLON", "Deklarasi harus diakhiri dengan ';'.")

        data_type = "int" if type_token.type == "INT_TYPE" else "bool"
        return Declaration(data_type, name, value)

    def parse_if_else(self) -> IfElse:
        self.consume("IF", "Statement harus diawali dengan keyword 'if'.")
        self.consume("LPAREN", "Kondisi if harus diawali dengan '('.")
        condition = self.parse_binary_expression()
        self.consume("RPAREN", "Kondisi if harus diakhiri dengan ')'.")
        self.consume("LBRACE", "Blok then harus diawali dengan '{'.")
        then_branch = self.parse_assignment()
        self.consume("RBRACE", "Blok then harus diakhiri dengan '}'.")
        self.consume("ELSE", "Statement if harus memiliki keyword 'else'.")
        self.consume("LBRACE", "Blok else harus diawali dengan '{'.")
        else_branch = self.parse_assignment()
        self.consume("RBRACE", "Blok else harus diakhiri dengan '}'.")
        return IfElse(condition, then_branch, else_branch)

    def parse_assignment(self) -> Assignment:
        name = self.consume("IDENTIFIER", "Assignment membutuhkan nama variabel.").value
        self.consume("ASSIGN", "Assignment membutuhkan operator '='.")
        expression = self.parse_expression()
        self.consume("SEMICOLON", "Assignment harus diakhiri dengan ';'.")
        return Assignment(name, expression)

    def parse_expression(self) -> Union[BinaryExpression, int, bool, str]:
        left = self.parse_operand()

        if self.peek().type == "OPERATOR":
            operator = self.advance().value
            right = self.parse_operand()
            return BinaryExpression(left, operator, right)

        return left

    def parse_binary_expression(self) -> BinaryExpression:
        expression = self.parse_expression()
        if not isinstance(expression, BinaryExpression):
            raise SyntaxError("Kondisi harus berupa ekspresi biner, misalnya: x > 5.")
        return expression

    def parse_operand(self) -> Union[int, bool, str]:
        if self.peek().type == "NUMBER":
            return int(self.advance().value)
        if self.peek().type in ("TRUE", "FALSE"):
            return self.advance().type == "TRUE"
        if self.peek().type == "IDENTIFIER":
            return self.advance().value
        raise SyntaxError(f"Operand tidak valid pada token '{self.peek().value}'.")

    def parse_literal(self) -> Union[int, bool]:
        if self.peek().type == "NUMBER":
            return int(self.advance().value)
        if self.peek().type in ("TRUE", "FALSE"):
            return self.advance().type == "TRUE"
        raise SyntaxError("Deklarasi hanya menerima literal angka atau boolean.")

    def consume(self, token_type: str, message: str) -> Token:
        if self.peek().type == token_type:
            return self.advance()
        raise SyntaxError(f"{message} Ditemukan '{self.peek().value}' pada posisi {self.peek().position}.")

    def advance(self) -> Token:
        token = self.peek()
        self.current += 1
        return token

    def peek(self) -> Token:
        return self.tokens[self.current]


class SemanticAnalyzer:
    def __init__(self):
        self.symbol_table: Dict[str, str] = {}

    def analyze(self, program: Program) -> Dict[str, str]:
        for declaration in program.declarations:
            if declaration.name in self.symbol_table:
                raise TypeError(f"Variabel '{declaration.name}' sudah dideklarasikan.")

            value_type = self.infer_type(declaration.value)
            if declaration.data_type != value_type:
                raise TypeError(
                    f"Tipe deklarasi '{declaration.name}' tidak cocok: "
                    f"diharapkan {declaration.data_type}, ditemukan {value_type}."
                )

            self.symbol_table[declaration.name] = declaration.data_type

        condition_type = self.infer_type(program.statement.condition)
        if condition_type != "bool":
            raise TypeError("Kondisi if harus bertipe bool.")

        self.validate_assignment(program.statement.then_branch)
        self.validate_assignment(program.statement.else_branch)
        return self.symbol_table

    def validate_assignment(self, assignment: Assignment) -> None:
        if assignment.name not in self.symbol_table:
            raise NameError(f"Variabel '{assignment.name}' belum dideklarasikan.")

        target_type = self.symbol_table[assignment.name]
        expression_type = self.infer_type(assignment.expression)

        if target_type != expression_type:
            raise TypeError(
                f"Assignment ke '{assignment.name}' tidak valid: "
                f"diharapkan {target_type}, ditemukan {expression_type}."
            )

    def infer_type(self, expression: Union[BinaryExpression, int, bool, str]) -> str:
        if isinstance(expression, bool):
            return "bool"
        if isinstance(expression, int):
            return "int"
        if isinstance(expression, str):
            if expression not in self.symbol_table:
                raise NameError(f"Variabel '{expression}' belum dideklarasikan.")
            return self.symbol_table[expression]

        left_type = self.infer_type(expression.left)
        right_type = self.infer_type(expression.right)

        if expression.operator in ("+", "-", "*", "/"):
            if left_type == right_type == "int":
                return "int"
            raise TypeError("Operator aritmetika hanya dapat digunakan pada tipe int.")

        if expression.operator in (">", "<", ">=", "<="):
            if left_type == right_type == "int":
                return "bool"
            raise TypeError("Operator pembanding urutan hanya dapat digunakan pada tipe int.")

        if expression.operator in ("==", "!="):
            if left_type == right_type:
                return "bool"
            raise TypeError("Operator kesetaraan membutuhkan operand dengan tipe yang sama.")

        raise TypeError(f"Operator '{expression.operator}' tidak didukung.")


class TACGenerator:
    def __init__(self):
        self.temp_counter = 1
        self.label_counter = 1
        self.instructions: List[str] = []

    def generate(self, program: Program) -> List[str]:
        for declaration in program.declarations:
            self.instructions.append(f"{declaration.name} = {self.format_operand(declaration.value)}")

        condition_place = self.emit_expression(program.statement.condition)
        else_label = self.new_label()
        end_label = self.new_label()

        self.instructions.append(f"ifFalse {condition_place} goto {else_label}")
        self.emit_assignment(program.statement.then_branch)
        self.instructions.append(f"goto {end_label}")
        self.instructions.append(f"{else_label}:")
        self.emit_assignment(program.statement.else_branch)
        self.instructions.append(f"{end_label}:")
        return self.instructions

    def emit_assignment(self, assignment: Assignment) -> None:
        expression_place = self.emit_expression(assignment.expression)
        self.instructions.append(f"{assignment.name} = {expression_place}")

    def emit_expression(self, expression: Union[BinaryExpression, int, bool, str]) -> str:
        if not isinstance(expression, BinaryExpression):
            return self.format_operand(expression)

        left = self.emit_expression(expression.left)
        right = self.emit_expression(expression.right)
        temp = self.new_temp()
        self.instructions.append(f"{temp} = {left} {expression.operator} {right}")
        return temp

    def format_operand(self, operand: Union[int, bool, str]) -> str:
        if isinstance(operand, bool):
            return "true" if operand else "false"
        return str(operand)

    def new_temp(self) -> str:
        temp = f"t{self.temp_counter}"
        self.temp_counter += 1
        return temp

    def new_label(self) -> str:
        label = f"L{self.label_counter}"
        self.label_counter += 1
        return label


class IfElseCompiler:
    def __init__(self, source_code: str):
        self.source_code = source_code
        self.tokens: List[Token] = []
        self.ast: Optional[Program] = None
        self.symbol_table: Dict[str, str] = {}
        self.tac: List[str] = []

    def compile(self) -> None:
        self.tokens = Lexer(self.source_code).tokenize()
        self.ast = Parser(self.tokens).parse()
        self.symbol_table = SemanticAnalyzer().analyze(self.ast)
        self.tac = TACGenerator().generate(self.ast)


def print_section(title: str, content: object) -> None:
    print(f"\n=== {title} ===")
    print(content)


def main() -> None:
    source_code = """
    int x = 8;
    int y = 0;

    if (x > 5) {
        y = 1;
    } else {
        y = 0;
    }
    """

    compiler = IfElseCompiler(source_code)
    compiler.compile()

    token_lines = [f"{token.type:<12} {token.value}" for token in compiler.tokens if token.type != "EOF"]

    print_section("Source Code", source_code.strip())
    print_section("Analisis Leksikal (Token)", "\n".join(token_lines))
    print_section("Analisis Sintaksis (AST)", compiler.ast)
    print_section("Analisis Semantik (Symbol Table)", compiler.symbol_table)
    print_section("Three-Address Code (TAC)", "\n".join(compiler.tac))


if __name__ == "__main__":
    main()
