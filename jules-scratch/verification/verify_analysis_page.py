from playwright.sync_api import sync_playwright, expect

def run_verification():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # 1. Navega diretamente para a URL da página de análise
        page.goto("http://localhost:8501/Analise_de_Desempenho")

        # Aguarda um pouco para a página carregar e os gráficos renderizarem
        page.wait_for_timeout(5000)

        # 2. Verifica se o título da página está correto
        expect(page.get_by_text("Análise de Acurácia Comparativa")).to_be_visible()

        # 3. Tira a captura de tela
        page.screenshot(path="jules-scratch/verification/analysis_page_v2.png")

        browser.close()

if __name__ == "__main__":
    run_verification()
