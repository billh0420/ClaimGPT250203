# parser03.py

from pathlib import Path

from shared.parsers.parser import Parser

from internal.frame_stack.FrameStack import FrameStack
from internal.proof.proof import Proof

class Parser03(Parser):

    def __init__(self, mmx_file_path: Path, limit_count: int or None):
        super().__init__(mm_file_path=mmx_file_path)
        self.count = 0
        self.frame_stack = FrameStack()
        self.labels = {}
        self.statements = []
        self.axiom_statements = []
        self.proved_statements = []
        self.label = None
        self.proved_statement_labels = set()
        self.used_proved_statement_labels = set()
        self.labelsByVariable = dict()
        self.result = dict()
        self._parse(limit_count=limit_count)
        self.tokens.close_file()  # Note this: kludge

    def _parse(self, limit_count: None or int):
        self.count = 0
        self.frame_stack = FrameStack()
        self.labels = {}
        self.statements = []
        self.axiom_statements = []
        self.proved_statements = []
        self.label = None
        self.proved_statement_labels = set()
        self.used_proved_statement_labels = set()
        self.frame_stack.push()
        done = False
        while not done:
            if limit_count and self.count > limit_count:
                break
            token = self.tokens.get_next_token()
            if token:
                self.count += 1
                if self.count % (1000000 // 100) == 0:
                    print(f'{self.count}: {token}')
                if token == '$a':
                    self._handle_axiom_statement()
                elif token == '$c':
                    self._handle_constant_statement()
                elif token == '$d':
                    self._handle_disjoint_statement()
                elif token == '$e':
                    self._handle_essential_hypothesis()
                elif token == '$f':
                    self._handle_floating_hypothesis()
                elif token == '$p':
                    self._handle_proved_statement()
                elif token == '$v':
                    self._handle_variable_statement()
                elif token == '$[':
                    self._handle_include_statement()
                elif token == '${':
                    self.frame_stack.push()
                elif token == '$}':
                    self.frame_stack.pop()
                else:
                    if not self.label:
                        self.label = token
                    else:
                        raise Exception(f'label not used: {self.label}')
            else:
                done = True
        print(self.tokens.token_buffer)

    def _handle_axiom_statement(self):
        if not self.label:
            raise Exception(f'No label for $a')
        math_symbols = []
        type_code = self.tokens.get_next_token()
        if not type_code:
            raise Exception(f'No type code for axiom statement')
        math_symbols.append(type_code)
        token = self.tokens.get_next_token()
        while token != '$.':
            math_symbols.append(token)
            token = self.tokens.get_next_token()
        if token:
            assertion = self.frame_stack.make_assertion(math_symbols)
            self.labels[self.label] = ('$a', assertion)
            axiom_statement = f'$a {self.label} {" ".join(math_symbols)} $.'
            self.axiom_statements.append(axiom_statement)
            self.statements.append(axiom_statement)
            if len(self.frame_stack[-1].e) > 0:
                f_hypotheses = [' '.join(x) for x in assertion[1]]
                e_hypotheses = [x.removeprefix('$e ').removesuffix(' $.') for x in self.frame_stack[-1].e_statements]
            else:
                f_hypotheses = [' '.join(x) for x in assertion[1]]
                e_hypotheses = []
            self.result[self.label] = [f_hypotheses, e_hypotheses, axiom_statement]
            self.label = None
        else:
            raise Exception('$a not closed')

    def _handle_constant_statement(self):
        if self.label:
            raise Exception(f'Label for $c')
        constants = []
        token = self.tokens.get_next_token()
        while token != '$.':
            constants.append(token)
            token = self.tokens.get_next_token()
        if token:
            for constant in constants:
                self.frame_stack.add_c(constant)
            constant_statement = f'$c {" ".join(constants)} $.'
            self.statements.append(constant_statement)
        else:
            raise Exception('$c not closed')

    def _handle_disjoint_statement(self):
        if self.label:
            raise Exception(f'Label for $d')
        variables = []
        token = self.tokens.get_next_token()
        if token:
            variables.append(token)
            token = self.tokens.get_next_token()
        else:
            raise Exception('$d not closed')
        if token:
            variables.append(token)
            token = self.tokens.get_next_token()
        else:
            raise Exception('$d not closed')
        while token != '$.':
            variables.append(token)
            token = self.tokens.get_next_token()
        if token:
            self.frame_stack.add_d(variables)
            disjoint_statement = f'$d {" ".join(variables)} $.'
            self.statements.append(disjoint_statement)
        else:
            raise Exception('$d not closed')

    def _handle_essential_hypothesis(self):
        if not self.label:
            raise Exception(f'No label for $e')
        math_symbols = []
        type_code = self.tokens.get_next_token()
        if not type_code:
            raise Exception(f'No type code for essential hypothesis')
        token = self.tokens.get_next_token()
        while token != '$.':
            math_symbols.append(token)
            token = self.tokens.get_next_token()
        if token:
            self.frame_stack.add_e(math_symbols, self.label)
            self.labels[self.label] = ('$e', f'{type_code} {" ".join(math_symbols)}')
            essential_hypothesis = f'$e {self.label} {type_code} {" ".join(math_symbols)} $.'
            self.statements.append(essential_hypothesis)
            if self.frame_stack:
                frame = self.frame_stack[-1]
                frame.e_statements.append(essential_hypothesis)
            self.label = None
        else:
            raise Exception('$e not closed')

    def _handle_floating_hypothesis(self):
        if not self.label:
            raise Exception(f'No label for $f')
        type_code = self.tokens.get_next_token()
        if not type_code:
            raise Exception('No type code for floating hypothesis')
        variable = self.tokens.get_next_token()
        if not variable:
            raise Exception('No variable for floating hypothesis')
        token = self.tokens.get_next_token()
        if token == '$.':
            self.frame_stack.add_f(variable, type_code, self.label)
            self.labels[self.label] = ('$f', [type_code, variable])
            floating_hypothesis = f'$f {self.label} {type_code} {variable} $.'
            self.statements.append(floating_hypothesis)
            self.label = None
        else:
            raise Exception('$f not closed')

    def _handle_proved_statement(self):
        if not self.label:
            raise Exception(f'No label for $p')
        math_symbols = []
        proof = []
        type_code = self.tokens.get_next_token()
        if not type_code:
            raise Exception(f'No type code for proved statement')
        math_symbols.append(type_code)
        token = self.tokens.get_next_token()
        while token != '$=':
            math_symbols.append(token)
            token = self.tokens.get_next_token()
        if token == '$=':
            token = self.tokens.get_next_token()
            while token != '$.':
                proof.append(token)
                token = self.tokens.get_next_token()
        if token:
            assertion = self.frame_stack.make_assertion(math_symbols)
            if proof[0] == '(':
                uncompressed_proof = Proof.decompress_proof_2(self.label, math_symbols, proof, self.frame_stack, self.labels)
            else:
                uncompressed_proof = proof
            proved_statement = f'$p {self.label} {" ".join(math_symbols)} $= {" ".join(uncompressed_proof)} $.'
            self.proved_statements.append(proved_statement)
            self.statements.append(proved_statement)
            self.labels[self.label] = ('$p', self.frame_stack.make_assertion(math_symbols))
            self.proved_statement_labels.add(self.label)
            for tau in uncompressed_proof:
                if tau in self.proved_statement_labels:
                    self.used_proved_statement_labels.add(tau)
            f_hypotheses = [' '.join(x) for x in assertion[1]]
            e_hypotheses = []
            for frame in self.frame_stack:
                if len(frame.e) > 0:
                    e_hypotheses += [x.removeprefix('$e ').removesuffix(' $.') for x in frame.e_statements]
            # if len(self.frame_stack[-1].e) > 0:
            #     e_hypotheses = [x.removeprefix('$e ').removesuffix(' $.') for x in self.frame_stack[-1].e_statements]
            # else:
            #     e_hypotheses = ""
            self.result[self.label] = [f_hypotheses, e_hypotheses, proved_statement]
            self.label = None
        else:
            raise Exception('$p not closed')

    def _handle_variable_statement(self):
        if self.label:
            raise Exception(f'Label for $v')
        variables = []
        token = self.tokens.get_next_token()
        while token != '$.':
            variables.append(token)
            token = self.tokens.get_next_token()
        if token:
            for variable in variables:
                self.frame_stack.add_v(variable)
            variable_statement = f'$v {" ".join(variables)} $.'
            self.statements.append(variable_statement)
        else:
            raise Exception('$v not closed')

    def _handle_include_statement(self):
        token = self.tokens.get_next_token()
        if token:
            filename = token
            token = self.tokens.get_next_token()
            if token == '$]':
                print(f'$[ {filename} $]')
            else:
                raise Exception('$[ not closed')
