from _typeshed import Incomplete
from typing import ClassVar, Final

from docutils import nodes
from docutils.writers import _html_base

__docformat__: Final = "reStructuredText"

class Writer(_html_base.Writer):
    default_stylesheets: ClassVar[list[str]]
    default_stylesheet_dirs: ClassVar[list[str]]
    default_template: ClassVar[str]
    config_section: ClassVar[str]
    translator_class: type[HTMLTranslator]

class HTMLTranslator(_html_base.HTMLTranslator):
    content_type: ClassVar[str]
    content_type_mathml: ClassVar[str]
    object_image_types: ClassVar[dict[str, str]]
    def set_first_last(self, node) -> None: ...
    def visit_address(self, node: nodes.address) -> None: ...
    def depart_address(self, node: nodes.address) -> None: ...
    def visit_admonition(self, node: nodes.admonition) -> None: ...
    def depart_admonition(self, node: nodes.admonition | None = None) -> None: ...
    def visit_author(self, node: nodes.author) -> None: ...
    author_in_authors: bool
    def depart_author(self, node: nodes.author) -> None: ...
    def visit_authors(self, node: nodes.authors) -> None: ...
    def depart_authors(self, node: nodes.authors) -> None: ...
    def visit_colspec(self, node: nodes.colspec) -> None: ...
    def depart_colspec(self, node: nodes.colspec) -> None: ...
    def is_compactable(self, node: nodes.Element) -> bool: ...
    def visit_citation(self, node: nodes.citation) -> None: ...
    def depart_citation(self, node: nodes.citation) -> None: ...
    def visit_citation_reference(self, node: nodes.citation_reference) -> None: ...
    def depart_citation_reference(self, node: nodes.citation_reference) -> None: ...
    def visit_classifier(self, node: nodes.classifier) -> None: ...
    def depart_classifier(self, node: nodes.classifier) -> None: ...
    def visit_compound(self, node: nodes.compound) -> None: ...
    def depart_compound(self, node: nodes.compound) -> None: ...
    def visit_definition(self, node: nodes.definition) -> None: ...
    def depart_definition(self, node: nodes.definition) -> None: ...
    def visit_definition_list(self, node: nodes.definition_list) -> None: ...
    def depart_definition_list(self, node: nodes.definition_list) -> None: ...
    def visit_definition_list_item(self, node: nodes.definition_list_item) -> None: ...
    def depart_definition_list_item(self, node: nodes.definition_list_item) -> None: ...
    def visit_description(self, node: nodes.description) -> None: ...
    def depart_description(self, node: nodes.description) -> None: ...
    in_docinfo: bool
    def visit_docinfo(self, node: nodes.docinfo) -> None: ...
    docinfo: Incomplete
    body: Incomplete
    def depart_docinfo(self, node: nodes.docinfo) -> None: ...
    def visit_docinfo_item(self, node, name, meta: bool = True) -> None: ...
    def depart_docinfo_item(self) -> None: ...
    def visit_doctest_block(self, node) -> None: ...
    def depart_doctest_block(self, node) -> None: ...
    def visit_entry(self, node) -> None: ...
    def depart_entry(self, node) -> None: ...
    compact_p: Incomplete
    compact_simple: Incomplete
    def visit_enumerated_list(self, node) -> None: ...
    def depart_enumerated_list(self, node) -> None: ...
    def visit_field(self, node) -> None: ...
    def depart_field(self, node) -> None: ...
    def visit_field_body(self, node) -> None: ...
    def depart_field_body(self, node) -> None: ...
    compact_field_list: bool
    def visit_field_list(self, node: nodes.field_list) -> None: ...
    def depart_field_list(self, node: nodes.field_list) -> None: ...
    def visit_field_name(self, node: nodes.field_name) -> None: ...
    def depart_field_name(self, node: nodes.field_name) -> None: ...
    def visit_footnote(self, node: nodes.footnote) -> None: ...
    def footnote_backrefs(self, node: nodes.footnote) -> None: ...
    def depart_footnote(self, node: nodes.footnote) -> None: ...
    def visit_footnote_reference(self, node: nodes.footnote_reference) -> None: ...
    def depart_footnote_reference(self, node: nodes.footnote_reference) -> None: ...
    def visit_generated(self, node: nodes.generated) -> None: ...
    def visit_image(self, node: nodes.image) -> None: ...
    def depart_image(self, node: nodes.image) -> None: ...
    def visit_label(self, node: nodes.label) -> None: ...
    def depart_label(self, node: nodes.label) -> None: ...
    def visit_list_item(self, node: nodes.list_item) -> None: ...
    def depart_list_item(self, node: nodes.list_item) -> None: ...
    def visit_literal(self, node: nodes.literal) -> None: ...
    def depart_literal(self, node: nodes.literal) -> None: ...
    def visit_literal_block(self, node: nodes.literal_block) -> None: ...
    def depart_literal_block(self, node: nodes.literal_block) -> None: ...
    def visit_option_group(self, node: nodes.option_group) -> None: ...
    def depart_option_group(self, node: nodes.option_group) -> None: ...
    def visit_option_list(self, node: nodes.option_list) -> None: ...
    def depart_option_list(self, node: nodes.option_list) -> None: ...
    def visit_option_list_item(self, node: nodes.option_list_item) -> None: ...
    def depart_option_list_item(self, node: nodes.option_list_item) -> None: ...
    def should_be_compact_paragraph(self, node: nodes.Element) -> bool: ...
    def visit_paragraph(self, node: nodes.paragraph) -> None: ...
    def depart_paragraph(self, node: nodes.paragraph) -> None: ...
    in_sidebar: bool
    def visit_sidebar(self, node: nodes.sidebar) -> None: ...
    def depart_sidebar(self, node: nodes.sidebar) -> None: ...
    def visit_subscript(self, node: nodes.subscript) -> None: ...
    def depart_subscript(self, node: nodes.subscript) -> None: ...
    in_document_title: int
    def visit_subtitle(self, node: nodes.subtitle) -> None: ...
    subtitle: list[Incomplete]
    def depart_subtitle(self, node: nodes.subtitle) -> None: ...
    def visit_superscript(self, node: nodes.superscript) -> None: ...
    def depart_superscript(self, node: nodes.superscript) -> None: ...
    def visit_system_message(self, node: nodes.system_message) -> None: ...
    def depart_system_message(self, node: nodes.system_message) -> None: ...
    def visit_table(self, node: nodes.table) -> None: ...
    def depart_table(self, node: nodes.table) -> None: ...
    def visit_tbody(self, node: nodes.tbody) -> None: ...
    def depart_tbody(self, node: nodes.tbody) -> None: ...
    def visit_term(self, node: nodes.term) -> None: ...
    def depart_term(self, node: nodes.term) -> None: ...
    def visit_thead(self, node: nodes.thead) -> None: ...
    def depart_thead(self, node: nodes.thead) -> None: ...
    def section_title_tags(self, node: nodes.Element) -> tuple[str, str]: ...

class SimpleListChecker(_html_base.SimpleListChecker):
    def visit_list_item(self, node: nodes.list_item) -> None: ...
    def visit_paragraph(self, node: nodes.paragraph) -> None: ...
    def visit_definition_list(self, node: nodes.definition_list) -> None: ...
    def visit_docinfo(self, node: nodes.docinfo) -> None: ...
